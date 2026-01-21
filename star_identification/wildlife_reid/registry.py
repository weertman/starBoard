"""
Dataset Registry for Wildlife ReID-10k

Each sub-dataset has unique characteristics that affect how it should be split
and evaluated. This registry captures those characteristics.
"""
from dataclasses import dataclass
from typing import Optional, List, Dict
from enum import Enum


class SplitStrategy(Enum):
    """Recommended split strategy for a dataset."""
    ORIGINAL = "original"       # Use provided train/test split
    TIME_AWARE = "time_aware"   # Split by temporal information
    CLUSTER_AWARE = "cluster_aware"  # Split by visual clusters
    RANDOM = "random"           # Random identity-level split


class BodyPart(Enum):
    """Primary body part visible in dataset images."""
    FULL_BODY = "full_body"
    FACE = "face"
    FLANK = "flank"
    BACK = "back"
    HEAD = "head"
    DORSAL = "dorsal"
    MIXED = "mixed"


@dataclass
class DatasetInfo:
    """Information about a specific sub-dataset in Wildlife10k."""
    
    name: str
    species: str
    
    # Data characteristics
    has_dates: bool = False
    has_clusters: bool = False
    has_orientation: bool = False
    
    # Recommended handling
    recommended_split: SplitStrategy = SplitStrategy.ORIGINAL
    body_part: BodyPart = BodyPart.FULL_BODY
    
    # Quality/difficulty indicators
    high_intraclass_variance: bool = False  # Same individual looks different
    low_interclass_variance: bool = False   # Different individuals look similar
    video_sourced: bool = False             # Images extracted from video (high similarity)
    
    # Size info (approximate, filled from actual data)
    approx_images: int = 0
    approx_identities: int = 0
    
    # Notes for special handling
    notes: Optional[str] = None
    
    def __repr__(self):
        return f"DatasetInfo({self.name}, {self.species}, split={self.recommended_split.value})"


class DatasetRegistry:
    """Registry of all Wildlife10k sub-datasets with their characteristics."""
    
    def __init__(self):
        self._datasets: Dict[str, DatasetInfo] = {}
        self._initialize_registry()
    
    def _initialize_registry(self):
        """Initialize registry with known dataset characteristics."""
        
        # ===== FISH =====
        self.register(DatasetInfo(
            name="AAUZebraFish",
            species="fish",
            has_clusters=True,
            video_sourced=True,
            body_part=BodyPart.FULL_BODY,
            recommended_split=SplitStrategy.CLUSTER_AWARE,
            notes="Zebrafish from video, high frame similarity"
        ))
        
        # ===== CATTLE/COWS =====
        self.register(DatasetInfo(
            name="AerialCattle2017",
            species="cow",
            has_clusters=True,
            body_part=BodyPart.BACK,
            recommended_split=SplitStrategy.CLUSTER_AWARE,
            notes="Aerial/drone imagery of cattle"
        ))
        
        self.register(DatasetInfo(
            name="CowDataset",
            species="cow",
            body_part=BodyPart.FULL_BODY,
            recommended_split=SplitStrategy.ORIGINAL,
        ))
        
        self.register(DatasetInfo(
            name="Cows2021",
            species="cow",
            body_part=BodyPart.FULL_BODY,
            recommended_split=SplitStrategy.ORIGINAL,
            notes="Large-scale cattle dataset"
        ))
        
        self.register(DatasetInfo(
            name="FriesianCattle2015",
            species="cow",
            body_part=BodyPart.FLANK,
            recommended_split=SplitStrategy.ORIGINAL,
            notes="Holstein-Friesian cattle coat patterns"
        ))
        
        self.register(DatasetInfo(
            name="FriesianCattle2017",
            species="cow",
            body_part=BodyPart.FLANK,
            recommended_split=SplitStrategy.ORIGINAL,
        ))
        
        self.register(DatasetInfo(
            name="MultiCamCows2024",
            species="cow",
            body_part=BodyPart.FULL_BODY,
            recommended_split=SplitStrategy.ORIGINAL,
            notes="Multi-camera setup for re-ID"
        ))
        
        self.register(DatasetInfo(
            name="OpenCows2020",
            species="cow",
            body_part=BodyPart.FULL_BODY,
            recommended_split=SplitStrategy.ORIGINAL,
        ))
        
        # ===== SEA TURTLES =====
        self.register(DatasetInfo(
            name="AmvrakikosTurtles",
            species="sea turtle",
            has_dates=True,
            recommended_split=SplitStrategy.TIME_AWARE,
            body_part=BodyPart.FACE,
            notes="Loggerhead sea turtles, face patterns"
        ))
        
        self.register(DatasetInfo(
            name="ReunionTurtles",
            species="sea turtle",
            body_part=BodyPart.FACE,
            recommended_split=SplitStrategy.ORIGINAL,
        ))
        
        self.register(DatasetInfo(
            name="SeaTurtleID2022",
            species="sea turtle",
            body_part=BodyPart.FACE,
            recommended_split=SplitStrategy.ORIGINAL,
            high_intraclass_variance=True,
            notes="Large-scale turtle face re-ID"
        ))
        
        self.register(DatasetInfo(
            name="SouthernProvinceTurtles",
            species="sea turtle",
            body_part=BodyPart.FACE,
            recommended_split=SplitStrategy.ORIGINAL,
        ))
        
        self.register(DatasetInfo(
            name="ZakynthosTurtles",
            species="sea turtle",
            body_part=BodyPart.FACE,
            recommended_split=SplitStrategy.ORIGINAL,
        ))
        
        self.register(DatasetInfo(
            name="ZindiTurtleRecall",
            species="sea turtle",
            body_part=BodyPart.FACE,
            recommended_split=SplitStrategy.ORIGINAL,
            notes="Competition dataset"
        ))
        
        # ===== BIG CATS =====
        self.register(DatasetInfo(
            name="ATRW",
            species="tiger",
            has_clusters=True,
            body_part=BodyPart.FULL_BODY,
            recommended_split=SplitStrategy.CLUSTER_AWARE,
            notes="Amur Tiger Re-ID in the Wild"
        ))
        
        self.register(DatasetInfo(
            name="LeopardID2022",
            species="leopard",
            body_part=BodyPart.FLANK,
            recommended_split=SplitStrategy.ORIGINAL,
            notes="Leopard spot patterns"
        ))
        
        # ===== MARINE MAMMALS =====
        self.register(DatasetInfo(
            name="BelugaID",
            species="whale",
            has_dates=True,
            has_clusters=True,
            recommended_split=SplitStrategy.TIME_AWARE,
            body_part=BodyPart.DORSAL,
            notes="Beluga whales, dorsal ridge scars"
        ))
        
        self.register(DatasetInfo(
            name="SealID",
            species="seal",
            body_part=BodyPart.FACE,
            recommended_split=SplitStrategy.ORIGINAL,
            notes="Harbor seal face patterns"
        ))
        
        self.register(DatasetInfo(
            name="PolarBearVidID",
            species="polar bear",
            video_sourced=True,
            body_part=BodyPart.FULL_BODY,
            recommended_split=SplitStrategy.CLUSTER_AWARE,
            notes="Polar bears from video"
        ))
        
        self.register(DatasetInfo(
            name="WhaleSharkID",
            species="whaleshark",
            body_part=BodyPart.FLANK,
            recommended_split=SplitStrategy.ORIGINAL,
            notes="Whale shark spot patterns"
        ))
        
        self.register(DatasetInfo(
            name="NDD20",
            species="dolphin",
            body_part=BodyPart.DORSAL,
            recommended_split=SplitStrategy.ORIGINAL,
            notes="Dolphin dorsal fin nicks"
        ))
        
        # ===== PRIMATES =====
        self.register(DatasetInfo(
            name="CTai",
            species="chimpanzee",
            has_clusters=True,
            body_part=BodyPart.FACE,
            recommended_split=SplitStrategy.CLUSTER_AWARE,
            notes="Taï chimpanzees"
        ))
        
        self.register(DatasetInfo(
            name="CZoo",
            species="chimpanzee",
            has_clusters=True,
            body_part=BodyPart.FACE,
            recommended_split=SplitStrategy.CLUSTER_AWARE,
            notes="Zoo chimpanzees"
        ))
        
        self.register(DatasetInfo(
            name="PrimFace",
            species="macaque",
            body_part=BodyPart.FACE,
            recommended_split=SplitStrategy.ORIGINAL,
            notes="Macaque face recognition"
        ))
        
        # ===== PETS =====
        self.register(DatasetInfo(
            name="CatIndividualImages",
            species="cat",
            body_part=BodyPart.FACE,
            recommended_split=SplitStrategy.ORIGINAL,
            low_interclass_variance=True,
            notes="Large cat face dataset, many similar cats"
        ))
        
        self.register(DatasetInfo(
            name="DogFaceNet",
            species="dog",
            body_part=BodyPart.FACE,
            recommended_split=SplitStrategy.ORIGINAL,
            notes="Dog face recognition"
        ))
        
        self.register(DatasetInfo(
            name="MPDD",
            species="dog",
            body_part=BodyPart.FULL_BODY,
            recommended_split=SplitStrategy.ORIGINAL,
            notes="Missing Pet Dog Dataset"
        ))
        
        # ===== AFRICAN WILDLIFE =====
        self.register(DatasetInfo(
            name="Giraffes",
            species="giraffe",
            body_part=BodyPart.FLANK,
            recommended_split=SplitStrategy.ORIGINAL,
            notes="Giraffe coat patterns"
        ))
        
        self.register(DatasetInfo(
            name="GiraffeZebraID",
            species="zebra",
            body_part=BodyPart.FLANK,
            recommended_split=SplitStrategy.ORIGINAL,
            notes="Combined giraffe and zebra stripes"
        ))
        
        self.register(DatasetInfo(
            name="HyenaID2022",
            species="hyena",
            body_part=BodyPart.FLANK,
            recommended_split=SplitStrategy.ORIGINAL,
            notes="Spotted hyena patterns"
        ))
        
        self.register(DatasetInfo(
            name="NyalaData",
            species="nyala",
            body_part=BodyPart.FLANK,
            recommended_split=SplitStrategy.ORIGINAL,
            notes="Nyala antelope stripes"
        ))
        
        self.register(DatasetInfo(
            name="StripeSpotter",
            species="zebra",
            body_part=BodyPart.FLANK,
            recommended_split=SplitStrategy.ORIGINAL,
            notes="Zebra stripe patterns"
        ))
        
        # ===== BIRDS =====
        self.register(DatasetInfo(
            name="BirdIndividualID",
            species="bird",
            has_dates=True,
            has_clusters=True,
            recommended_split=SplitStrategy.TIME_AWARE,
            body_part=BodyPart.FULL_BODY,
            notes="Various bird species"
        ))
        
        self.register(DatasetInfo(
            name="Chicks4FreeID",
            species="chicken",
            body_part=BodyPart.FULL_BODY,
            recommended_split=SplitStrategy.ORIGINAL,
            notes="Chicken identification"
        ))
        
        # ===== PANDAS =====
        self.register(DatasetInfo(
            name="IPanda50",
            species="panda",
            body_part=BodyPart.FACE,
            recommended_split=SplitStrategy.ORIGINAL,
            high_intraclass_variance=True,
            notes="Giant panda face recognition"
        ))
        
        # ===== SEA STARS =====
        self.register(DatasetInfo(
            name="SeaStarReID2023",
            species="sea star",
            body_part=BodyPart.DORSAL,
            recommended_split=SplitStrategy.ORIGINAL,
            notes="Sunflower sea star identification - directly relevant to our project!"
        ))
        
        # ===== SYNTHETIC =====
        self.register(DatasetInfo(
            name="SMALST",
            species="zebra",  # Synthetic zebras
            body_part=BodyPart.FULL_BODY,
            recommended_split=SplitStrategy.ORIGINAL,
            notes="Synthetic 3D rendered zebras"
        ))
    
    def register(self, info: DatasetInfo):
        """Register a dataset."""
        self._datasets[info.name] = info
    
    def get(self, name: str) -> Optional[DatasetInfo]:
        """Get dataset info by name."""
        return self._datasets.get(name)
    
    def get_all(self) -> List[DatasetInfo]:
        """Get all registered datasets."""
        return list(self._datasets.values())
    
    def get_by_species(self, species: str) -> List[DatasetInfo]:
        """Get all datasets for a species."""
        return [d for d in self._datasets.values() if d.species == species]
    
    def get_by_split_strategy(self, strategy: SplitStrategy) -> List[DatasetInfo]:
        """Get all datasets with a specific recommended split."""
        return [d for d in self._datasets.values() if d.recommended_split == strategy]
    
    def get_time_aware_datasets(self) -> List[str]:
        """Get names of datasets that support time-aware splitting."""
        return [d.name for d in self._datasets.values() if d.has_dates]
    
    def get_cluster_aware_datasets(self) -> List[str]:
        """Get names of datasets that have cluster information."""
        return [d.name for d in self._datasets.values() if d.has_clusters]
    
    def list_species(self) -> List[str]:
        """Get unique species in the registry."""
        return sorted(set(d.species for d in self._datasets.values()))
    
    def summary(self) -> str:
        """Print summary of registered datasets."""
        lines = ["Wildlife10k Dataset Registry", "=" * 40]
        
        by_species = {}
        for d in self._datasets.values():
            if d.species not in by_species:
                by_species[d.species] = []
            by_species[d.species].append(d.name)
        
        for species in sorted(by_species.keys()):
            datasets = by_species[species]
            lines.append(f"\n{species.title()} ({len(datasets)} datasets):")
            for name in sorted(datasets):
                info = self._datasets[name]
                flags = []
                if info.has_dates:
                    flags.append("T")  # Time
                if info.has_clusters:
                    flags.append("C")  # Cluster
                if info.video_sourced:
                    flags.append("V")  # Video
                flag_str = f"[{','.join(flags)}]" if flags else ""
                lines.append(f"  - {name} {flag_str}")
        
        return "\n".join(lines)


# Global registry instance
DATASET_REGISTRY = DatasetRegistry()


