# Backend Archive TODO

This is the plain working checklist for the backend archive project.

Current direction:

- `Backblaze B2` is the primary cloud storage for images and other large blobs.
- The old multi-drive computer is the local backup mirror.
- Each workstation keeps a local working archive.
- A future backend will handle metadata, identity uniqueness, auth, and sync.

Do these in order. Do not skip ahead unless the earlier step is clearly done.

## Definition Of Done

- [ ] `Backblaze B2` is the primary blob store.
- [ ] The old computer is configured as a reliable backup mirror.
- [ ] Workstations do not need a full image mirror.
- [ ] Identity-code uniqueness is enforced centrally.
- [ ] Shared metadata and decisions have a sync path.
- [ ] There is a tested restore path from both cloud and local backup.

## Phase 1: Decide What Is Shared

### 1. Map all current archive state

- [ ] Create `docs/archive-authority-matrix.md`
- [ ] List every mutable archive artifact and where it lives now.
- [ ] For each item, record:
  - current location
  - current source of truth
  - future source of truth
  - whether it should be shared or local-only
- [ ] Make sure the table includes:
  - metadata CSVs
  - `_second_order_labels.csv`
  - silence markers
  - best-photo sidecars
  - merge history
  - metadata history
  - vocabularies
  - query pins
  - batch undo/redo history
  - DL registry
  - reports
  - logs
- Verify:
  - [ ] Nothing important in the current archive has been left out.

### 2. Freeze v1 scope

- [ ] Create `docs/shared-archive-v1-scope.md`
- [ ] Mark every item from the authority matrix as one of:
  - `shared in v1`
  - `local-only in v1`
  - `deferred`
- [ ] Explicitly list what is out of scope for v1.
- [ ] Explicitly list what remains local-only in v1.
- Verify:
  - [ ] No one implementing v1 would need to guess whether a file/state type is shared.

### 3. Freeze structural operation rules

- [ ] Create `docs/structural-operations-policy.md`
- [ ] Decide which of these must be online-only in v1:
  - create permanent identity code
  - rename identity
  - merge query into gallery
  - promote query to gallery
  - delete shared records
  - modify shared vocabularies
- [ ] Write a short reason for each rule.
- Verify:
  - [ ] A developer can tell which actions must go through the future backend.

## Phase 2: Storage And Backup

### 4. Define the `Backblaze B2` object model

- [ ] Create `docs/b2-blob-model.md`
- [ ] Define:
  - bucket names
  - object key format
  - asset IDs
  - checksum rules
  - deletion/versioning policy
- [ ] Add worked examples for:
  - gallery image
  - query image
  - morphometric asset
  - derived preview or thumbnail
- Verify:
  - [ ] Two different people would generate the same key for the same asset.

### 5. Build the backup computer

- [ ] Create `docs/backup-box-build.md`
- [ ] Decide the drive layout.
- [ ] Configure mirrored storage or equivalent redundancy.
- [ ] Document the filesystem/storage layout.
- [ ] Enable drive health monitoring.
- [ ] Document remote access.
- [ ] Document reboot/startup behavior.
- Verify:
  - [ ] The machine can reboot and come back with storage healthy and reachable.

### 6. Create the backup and restore runbook

- [ ] Create `docs/backup-runbook.md`
- [ ] Define the backup schedule from `Backblaze B2` to the local backup computer.
- [ ] Define retention or snapshot behavior.
- [ ] Define checksum verification.
- [ ] Write restore steps.
- [ ] Run at least one sample restore drill.
- Verify:
  - [ ] A sample object can be restored from:
    - `Backblaze B2`
    - the local backup computer
  - [ ] The restored object matches the original checksum.

## Phase 3: Backend Design

### 7. Design the control-plane schema

- [ ] Create `docs/control-plane-schema.md`
- [ ] Define where these will live in the future backend:
  - identities
  - human-facing codes
  - immutable internal IDs
  - encounters
  - metadata
  - decisions
  - device records
  - sync cursors/manifests
  - structural operation history
- [ ] Write down uniqueness constraints.
- [ ] Write down versioning fields.
- Verify:
  - [ ] Every `shared in v1` item from the scope doc has a home in the backend design.

### 8. Design device enrollment and auth

- [ ] Create `docs/device-enrollment.md`
- [ ] Define:
  - how a new workstation is approved
  - where credentials are stored
  - how a device is revoked
  - how credentials rotate
  - what happens for a lost or stolen device
- [ ] Decide whether access uses:
  - direct HTTPS
  - `Tailscale`
  - or both
- Verify:
  - [ ] A developer can describe exactly how a computer becomes authorized and unauthorized.

### 9. Write the sync contract

- [ ] Create `docs/sync-contract.md`
- [ ] Define:
  - local manifest format
  - revision/version rules
  - conflict classes
  - pull behavior
  - push behavior
- [ ] Include worked examples for:
  - pull selected images
  - push a new upload
  - edit metadata offline then sync
  - duplicate identity-code attempt
  - rename or merge as online-only
- Verify:
  - [ ] A developer can implement one sync flow without inventing missing rules.

### 10. Define local archive materialization

- [ ] Create `docs/local-archive-materialization.md`
- [ ] Define how remote shared state appears inside each workstation's local
      `archive/`.
- [ ] Define:
  - how pulled images land on disk
  - how metadata is materialized locally
  - what happens when metadata exists but an image has not been pulled
  - how local-only files behave in v1
- Verify:
  - [ ] A reviewer can tell exactly what will exist on disk after a partial pull.

## Phase 4: Prove The Design

### 11. Build a blob sync prototype

- [ ] Build a minimal prototype for selective image sync with `Backblaze B2`.
- [ ] Show that:
  - machine A can upload a sample encounter
  - machine B can pull only a selected subset
  - machine B does not pull the full library
  - checksums match after pull
- [ ] Write a short verification note or demo transcript.
- Verify:
  - [ ] The before/after local archive contents are shown for both machines.

### 12. Build a metadata and identity prototype

- [ ] Build a minimal prototype for:
  - central identity-code reservation
  - metadata sync
  - decision sync
- [ ] Show that:
  - duplicate identity codes are rejected centrally
  - a metadata update on one machine syncs to another
  - a saved decision on one machine syncs to another
- [ ] Write a short verification note or demo transcript.
- Verify:
  - [ ] The duplicate-code failure and successful cross-machine sync are both documented.

## Phase 5: Pilot And Recovery

### 13. Run a two-machine pilot

- [ ] Create `docs/pilot-results.md`
- [ ] Run a real workflow with two workstations.
- [ ] Confirm:
  - selective image pull works
  - metadata sync works
  - the backup mirror stays current
- [ ] Record:
  - what worked
  - what failed
  - what changed in the plan
- Verify:
  - [ ] Another maintainer can read the pilot doc and understand what is ready and what is not.

### 14. Run a disaster recovery drill

- [ ] Create `docs/disaster-recovery-drill.md`
- [ ] Restore a sample object from `Backblaze B2`.
- [ ] Restore a sample object from the local backup computer.
- [ ] Restore metadata or backend backups.
- [ ] Write the restore order clearly.
- Verify:
  - [ ] Another maintainer can follow the document and repeat the recovery drill.

## Explicit V1 Decisions Still To Make

- [ ] Decide whether `query pins` remain local-only in v1.
- [ ] Decide whether `batch undo/redo` remains local-only in v1.
- [ ] Decide whether `DL precompute` remains local-only in v1.
- [ ] Decide whether `reports` remain local-only in v1.
- [ ] Decide whether `interaction logs` remain local-only in v1.
- [ ] Decide whether shared vocabularies are in v1 or deferred.

## Recommended Order

- [ ] Finish steps 1-3 first
- [ ] Finish steps 4-6 second
- [ ] Finish steps 7-10 third
- [ ] Finish steps 11-14 last

## Final Closeout Checklist

- [ ] Authority matrix exists
- [ ] V1 scope is frozen
- [ ] Structural operation rules are frozen
- [ ] `Backblaze B2` object model exists
- [ ] Backup computer is configured
- [ ] Backup runbook exists and was tested
- [ ] Control-plane schema exists
- [ ] Device enrollment/auth design exists
- [ ] Sync contract exists
- [ ] Local archive materialization exists
- [ ] Blob sync prototype succeeded
- [ ] Metadata sync prototype succeeded
- [ ] Two-machine pilot completed
- [ ] Disaster recovery drill completed
