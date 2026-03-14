# Shared Archive Plan for `starBoard`

**Status:** Recommended direction  
**Date:** 2026-03-10  
**Audience:** Maintainers and collaborators

## Summary

`starBoard` is currently a local-first desktop application that reads and writes
directly to a filesystem archive rooted at `archive/` or a custom path provided
through `STARBOARD_ARCHIVE_DIR`.

We want to evolve that archive into a shared multi-computer system without
forcing every workstation to keep a full copy of all images. We also want to
keep costs low, preserve local responsiveness, allow explicit sync, and prevent
identity-code reuse.

The recommended deployment is:

1. **Backblaze B2 as the primary cloud image/blob store**
2. **A local multi-drive backup computer as a secondary backup mirror**
3. **A local working archive on each workstation**
4. **A future central metadata/API layer for identity control, sync, and
   authentication**

This gives us cheap storage, offsite durability, a practical local backup, and
room to grow into a proper shared archive without pretending a shared folder is
the same thing as a real backend.

## Chosen Direction

### Primary storage

Use **Backblaze B2** as the primary server-side storage for original images and
other large archive blobs.

Why this was chosen:

- low storage cost
- S3-compatible API
- good fit for archive and backup workflows
- no 90-day minimum retention penalty like some competitors
- good long-term path if `starBoard` gains a sync service

### Local backup

Use an older computer with **multiple internal or attached drives** as a
**local backup mirror**, not as the only copy of the archive.

That machine is useful because:

- it gives us a fast local restore target
- it provides a second copy under our control
- it can hold archive exports, snapshots, or a synced mirror
- it can later host lightweight services if needed

That machine should **not** be treated as the sole source of truth for the
shared archive. The primary copy should still be stored offsite in `Backblaze
B2`.

## High-Level Architecture

The target system should have four layers:

1. **Workstations**
   - Each user computer keeps a local working archive.
   - Each computer stores only the images it has explicitly pulled.
   - Metadata should eventually sync broadly; images should sync selectively.

2. **Primary cloud blob storage**
   - `Backblaze B2`
   - Stores original uploaded images and other large immutable blobs
   - Acts as the durable offsite archive

3. **Local backup mirror**
   - The old multi-drive computer
   - Holds a local replicated copy of important archive data
   - Used for backup validation, local restores, and disaster recovery

4. **Future metadata/auth/sync service**
   - A small API plus database
   - Handles identity registration, machine authentication, metadata sync,
     decisions, and structural operations like merge/rename/promote

## What Lives Where

### On each workstation

Each workstation should keep:

- a local working archive
- the subset of images that user explicitly pulled
- local caches and derived files
- local DL caches and precompute outputs
- local reports and temporary exports

This preserves the current desktop-first `starBoard` behavior and keeps the UI
responsive.

### In `Backblaze B2`

`Backblaze B2` should eventually hold:

- original uploaded images
- optional thumbnails or derived previews
- optional morphometric raw exports or packages
- compressed archive exports
- offsite backups of metadata/database dumps

### On the local backup computer

The backup computer should hold:

- a mirrored backup of the cloud archive
- regular snapshots or versioned backups if possible
- exported metadata backups
- restore-ready copies of high-value data

It may also later host:

- a small metadata API
- a local PostgreSQL database
- scheduled sync and backup jobs

But that hosting role is secondary. The first job of the old computer is
**backup and recovery**.

## Recommended Backup Strategy

Use the old multi-drive computer as a **backup mirror** with drive redundancy.

Recommended minimum setup:

- at least 2 large drives
- mirrored storage or equivalent redundancy
- automated nightly or scheduled sync from cloud storage
- regular verification that files can actually be restored

Recommended backup pattern:

1. Workstations push or sync archive data to the shared system
2. Primary archive blobs live in `Backblaze B2`
3. The old computer pulls a mirrored backup on a schedule
4. Metadata or database exports are also copied to the old computer
5. Periodic restore tests confirm the backup is usable

This gives:

- one offsite primary copy
- one local backup copy
- local workstation working copies for active use

## Why Not Use the Old Computer as the Only Server

It is tempting to make the old computer the whole backend, but that creates
avoidable risk.

Main concerns:

- home or office internet outages
- slower upstream bandwidth for remote users
- power loss
- aging disks or aging hardware
- more manual maintenance burden
- no offsite protection if the machine is stolen, damaged, or fails badly

The old computer is still very valuable, but it is better used as a **local
backup mirror and optional service host** rather than the only archive.

## Why This Fits `starBoard`

This plan fits the current codebase because `starBoard` is still fundamentally a
local archive application. It expects a local archive root and local file I/O.

That means the best near-term shared design is:

- keep local working archives
- sync data into those local archives explicitly
- avoid pretending an internet-mounted drive is the same thing as a fast local
  archive

This also matches the project goals:

- cost matters more than speed
- work should stay local-first
- users should explicitly choose what images to pull
- not every computer should have the entire archive

## Identity and Metadata Implications

The storage decision does **not** remove the need for a future metadata layer.

To properly support multiple computers, the project will still need a central
authority for:

- unique identity code registration
- machine authentication
- conflict-aware sync
- shared decisions and metadata state
- structural operations such as rename, merge, promote, and delete

So the chosen storage plan is:

- **cloud object storage for large files**
- **local mirrored backup for recovery**
- **future API/database for shared metadata control**

## Suggested Practical Deployment

The recommended practical deployment is:

- **Primary blob storage:** `Backblaze B2`
- **Local backup mirror:** old computer with multiple drives
- **Workstation archive model:** local working copies with selective sync
- **Future control plane:** small metadata API and database

If the team wants the simplest next step, the order should be:

1. Decide the storage bucket layout in `Backblaze B2`
2. Set up the old computer as a multi-drive backup machine
3. Define a backup schedule from cloud to local mirror
4. Design the metadata and identity service separately

## Storage Vendor Decision

After comparing typical low-cost object storage options, the recommended vendor
is:

### `Backblaze B2`

Reasons:

- low storage cost
- strong fit for archive-style data
- S3-compatible
- simpler long-term backend choice than trying to build around consumer sync
  products

Alternatives were considered, but not chosen as the primary recommendation:

- `Cloudflare R2`
  - attractive if egress becomes the main cost
  - higher storage cost than `B2`

- `IDrive e2`
  - very cheap headline pricing
  - less common choice for custom application backends

- `Wasabi`
  - decent storage price
  - less attractive here because of minimum retention behavior and policy fit
    concerns for some workloads

## Operational Notes

The backup computer should be treated like infrastructure, not like a casual
desktop.

Recommended practices:

- keep it on a UPS if possible
- use health monitoring for the drives
- use scheduled backup jobs
- keep system updates current
- use remote access through a private network layer such as `Tailscale`
- document restore steps

If it later hosts the metadata API, separate that from the backup process as
cleanly as possible.

## Final Recommendation

The chosen direction for the shared archive is:

- **Use `Backblaze B2` as the primary shared storage for images and large blobs**
- **Use the old multi-drive computer as a local backup mirror**
- **Keep `starBoard` local-first on each workstation**
- **Plan for a future metadata/API service for identity control, sync, and
  authentication**

This is the most practical balance of:

- low cost
- archive durability
- local restore capability
- future flexibility
- compatibility with how `starBoard` currently works
