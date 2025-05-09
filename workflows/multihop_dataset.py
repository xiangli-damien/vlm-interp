"""
Dataset utilities for multi-hop reasoning experiments.
Provides tools for loading, creating and managing two-hop prompt datasets.
"""

import os
import json
import random
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union
import logging

from workflows.multihop_vlm_experiments import TwoHopSample

# Configure logging
logger = logging.getLogger("multihop_dataset")
logger.setLevel(logging.INFO)

class MultihopDatasetLoader:
    """Utility for loading and processing multi-hop datasets."""
    
    @staticmethod
    def load_jsonl(path: str, max_samples: Optional[int] = None) -> List[TwoHopSample]:
        """
        Load a JSONL dataset file containing two-hop samples.
        
        Args:
            path: Path to JSONL file
            max_samples: Maximum number of samples to load
            
        Returns:
            List of TwoHopSample instances
        """
        samples = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_samples is not None and i >= max_samples:
                        break
                    
                    try:
                        data = json.loads(line.strip())
                        sample = TwoHopSample(**data)
                        samples.append(sample)
                    except Exception as e:
                        logger.warning(f"Error parsing line {i}: {e}")
                        continue
        except Exception as e:
            logger.error(f"Error loading dataset from {path}: {e}")
            raise
        
        logger.info(f"Loaded {len(samples)} samples from {path}")
        return samples
    
    @staticmethod
    def load_json(path: str, max_samples: Optional[int] = None) -> List[TwoHopSample]:
        """
        Load a JSON dataset file containing two-hop samples.
        
        Args:
            path: Path to JSON file
            max_samples: Maximum number of samples to load
            
        Returns:
            List of TwoHopSample instances
        """
        samples = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Handle list format
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        if max_samples is not None and i >= max_samples:
                            break
                        
                        try:
                            sample = TwoHopSample(**item)
                            samples.append(sample)
                        except Exception as e:
                            logger.warning(f"Error parsing item {i}: {e}")
                            continue
                
                # Handle dict format with 'samples' key
                elif isinstance(data, dict) and 'samples' in data:
                    for i, item in enumerate(data['samples']):
                        if max_samples is not None and i >= max_samples:
                            break
                        
                        try:
                            sample = TwoHopSample(**item)
                            samples.append(sample)
                        except Exception as e:
                            logger.warning(f"Error parsing item {i}: {e}")
                            continue
                
                else:
                    logger.error(f"Unknown JSON format in {path}")
                    raise ValueError(f"Unknown JSON format in {path}")
        
        except Exception as e:
            logger.error(f"Error loading dataset from {path}: {e}")
            raise
        
        logger.info(f"Loaded {len(samples)} samples from {path}")
        return samples
    
    @staticmethod
    def save_samples(samples: List[TwoHopSample], path: str, format: str = 'jsonl') -> None:
        """
        Save samples to a file.
        
        Args:
            samples: List of samples to save
            path: Output file path
            format: Output format ('json' or 'jsonl')
        """
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            
            # Convert samples to dictionaries
            sample_dicts = [asdict(sample) for sample in samples]
            
            if format.lower() == 'jsonl':
                with open(path, 'w', encoding='utf-8') as f:
                    for sample in sample_dicts:
                        f.write(json.dumps(sample) + '\n')
            
            elif format.lower() == 'json':
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(sample_dicts, f, indent=2)
            
            else:
                logger.error(f"Unknown format: {format}")
                raise ValueError(f"Unknown format: {format}")
            
            logger.info(f"Saved {len(samples)} samples to {path}")
        
        except Exception as e:
            logger.error(f"Error saving samples to {path}: {e}")
            raise

class MultihopSampleGenerator:
    """Utility for generating multi-hop samples."""
    
    @staticmethod
    def create_eiffel_tower_sample() -> TwoHopSample:
        """
        Create a sample for Eiffel Tower -> Seine River -> Famous Bridges example.
        
        Returns:
            TwoHopSample instance
        """
        return TwoHopSample(
            image="sample_images/eiffel_tower.jpg",
            prompt_two_hop="Name one of the famous bridges on the river next to the landmark in the picture.",
            prompt_one_hop="Name one of the famous bridges on the Seine River.",
            prompt_two_hop_entity_sub="Name one of the famous bridges on the river next to the Statue of Liberty.",
            prompt_two_hop_rel_sub="Name one of the famous restaurants on the river next to the landmark in the picture.",
            bridge_entity="Seine",
            bridge_entity_tokens=[]  # Will be filled by the experiment
        )
    
    @staticmethod
    def create_landmark_sample(
        image_path: str,
        landmark: str,
        nearby_feature: str,
        attribute: str,
        sub_landmark: str,
        sub_relation: str
    ) -> TwoHopSample:
        """
        Create a sample for landmark -> nearby feature -> attribute pattern.
        
        Args:
            image_path: Path to image
            landmark: Main landmark in the image
            nearby_feature: Feature near the landmark (bridge entity)
            attribute: Target attribute to query
            sub_landmark: Alternative landmark for entity substitution
            sub_relation: Alternative relation for relation substitution
            
        Returns:
            TwoHopSample instance
        """
        return TwoHopSample(
            image=image_path,
            prompt_two_hop=f"What is the {attribute} of the {nearby_feature} near the {landmark} in the picture?",
            prompt_one_hop=f"What is the {attribute} of the {nearby_feature}?",
            prompt_two_hop_entity_sub=f"What is the {attribute} of the {nearby_feature} near the {sub_landmark} in the picture?",
            prompt_two_hop_rel_sub=f"What is the {sub_relation} of the {nearby_feature} near the {landmark} in the picture?",
            bridge_entity=nearby_feature,
            bridge_entity_tokens=[]  # Will be filled by the experiment
        )