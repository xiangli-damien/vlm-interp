"""
Script to run multi-hop reasoning experiments on vision-language models.

Example usage:
    python -m workflows.run_multihop_experiment \
        --model_id llava-hf/llava-v1.6-mistral-7b-hf \
        --dataset_path data/multihop_samples.json \
        --output_dir results/multihop_experiment \
        --max_samples 10
"""

import os
import argparse
import logging
import torch
from typing import Dict, Any, List, Optional
import json
from workflows.multihop_vlm_experiments import MultihopVLMExperiment, ExperimentConfig, TwoHopSample
from workflows.multihop_dataset import MultihopDatasetLoader, MultihopSampleGenerator
from analysis.multihop_viz import plot_layer_statistics, plot_detailed_entrec, create_layer_heatmap
from runtime.model_utils import load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_multihop")

def create_default_samples() -> List[TwoHopSample]:
    """Create default samples if no dataset is provided."""
    generator = MultihopSampleGenerator()
    
    samples = [
        generator.create_eiffel_tower_sample(),
        generator.create_landmark_sample(
            image_path="sample_images/statue_of_liberty.jpg",
            landmark="Statue of Liberty",
            nearby_feature="harbor",
            attribute="name",
            sub_landmark="Eiffel Tower",
            sub_relation="depth"
        ),
        generator.create_landmark_sample(
            image_path="sample_images/colosseum.jpg",
            landmark="Colosseum",
            nearby_feature="forum",
            attribute="purpose",
            sub_landmark="Taj Mahal",
            sub_relation="color"
        )
    ]
    
    return samples

def run_experiments(args) -> Dict[str, Any]:
    """Run the experiments with the specified configuration."""
    # Create config
    config = ExperimentConfig(
        model_id=args.model_id,
        output_dir=args.output_dir,
        use_flash_attn=args.flash_attn,
        load_in_4bit=args.bits4,
        intervention_alpha=args.alpha,
        cache_dir=args.cache_dir,
        device=args.device
    )
    
    # Set device
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    
    # Load dataset
    if args.dataset_path and os.path.exists(args.dataset_path):
        if args.dataset_path.endswith(".jsonl"):
            samples = MultihopDatasetLoader.load_jsonl(
                args.dataset_path, 
                max_samples=args.max_samples
            )
        else:
            samples = MultihopDatasetLoader.load_json(
                args.dataset_path, 
                max_samples=args.max_samples
            )
    else:
        logger.warning(f"Dataset path {args.dataset_path} not found, using default samples")
        samples = create_default_samples()
    
    logger.info(f"Loaded {len(samples)} samples")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model {config.model_id}")
    model, processor = load_model(
        model_id=config.model_id,
        use_flash_attn=config.use_flash_attn,
        load_in_4bit=config.load_in_4bit,
        enable_gradients=True,
        device_map="auto" if device.type == "cuda" else None
    )
    
    # Create experiment
    experiment = MultihopVLMExperiment(model, processor, config, device)
    
    # Run experiments
    logger.info("Running experiments on samples...")
    results = experiment.run_experiment_batch(samples)
    
    # Save results
    results_path = os.path.join(config.output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(experiment._make_serializable(results), f, indent=2)
    logger.info(f"Saved results to {results_path}")
    
    # Create visualizations
    logger.info("Creating visualizations...")
    viz_dir = os.path.join(config.output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Layer statistics plot
    stats_plot = plot_layer_statistics(
        results, 
        os.path.join(viz_dir, "layer_statistics.png"),
        f"VLM Multi-hop Reasoning Analysis: {config.model_id.split('/')[-1]}"
    )
    logger.info(f"Saved layer statistics plot to {stats_plot}")
    
    # Layer heatmap
    heatmap = create_layer_heatmap(
        results,
        os.path.join(viz_dir, "layer_heatmap.png")
    )
    logger.info(f"Saved layer heatmap to {heatmap}")
    
    # Case study for first sample if available
    if samples and len(samples) > 0 and args.case_study:
        logger.info("Running detailed case study on first sample...")
        case_study = experiment.analyze_case_study(samples[0])
        
        # Save case study
        case_study_path = os.path.join(config.output_dir, "case_study.json")
        with open(case_study_path, 'w') as f:
            json.dump(experiment._make_serializable(case_study), f, indent=2)
        logger.info(f"Saved case study to {case_study_path}")
        
        # Plot detailed ENTREC values
        if "entrec_values" in case_study:
            entrec_plot = plot_detailed_entrec(
                case_study["entrec_values"],
                experiment.layers,
                case_study["bridge_entity"],
                os.path.join(viz_dir, "detailed_entrec.png")
            )
            logger.info(f"Saved detailed ENTREC plot to {entrec_plot}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-hop VLM experiments")
    
    # Model configuration
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf",
                        help="Hugging Face model ID")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory for model cache")
    parser.add_argument("--flash_attn", action="store_true",
                        help="Use Flash Attention 2 if available")
    parser.add_argument("--bits4", action="store_true",
                        help="Load model in 4-bit precision")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cuda, cpu)")
    
    # Dataset configuration
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to dataset file (JSON or JSONL)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    
    # Experiment configuration
    parser.add_argument("--output_dir", type=str, default="results/multihop_experiment",
                        help="Output directory for results and visualizations")
    parser.add_argument("--alpha", type=float, default=1e-2,
                        help="Step size for gradient intervention")
    parser.add_argument("--case_study", action="store_true",
                        help="Run detailed case study on first sample")
    
    args = parser.parse_args()
    results = run_experiments(args)