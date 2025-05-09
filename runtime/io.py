# runtime/io.py
"""
I/O utilities for data persistence and loading.
Enhanced for compatibility with older EnhancedSemanticTracer format.
"""

import os
import json
import pandas as pd
import torch
import csv
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logger = logging.getLogger("io_utils")
logger.setLevel(logging.INFO)

class TraceIO:
    """
    Input/Output handler for interpretability data.
    Manages CSV exports, metadata, and data loading with backward compatibility.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the I/O handler.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        # Use csv_data for compatibility with the older EnhancedSemanticTracer
        self.csv_dir = os.path.join(output_dir, "csv_data")
        self.meta_dir = os.path.join(output_dir, "meta")
        self.parquet_dir = os.path.join(output_dir, "parquet")
        
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.parquet_dir, exist_ok=True)
        
    def write_trace_data(self, trace_id: str, records: List[Dict[str, Any]], 
                     metadata: Dict[str, Any], format: str = "csv") -> str:
        """
        Write trace data to file and metadata to JSON.
        
        Args:
            trace_id: Unique identifier for the trace
            records: List of trace record dictionaries
            metadata: Dictionary of trace metadata
            format: Output format ("csv" or "parquet")
            
        Returns:
            Path to the created data file
        """
        if format.lower() == "csv":
            path = self._write_csv(trace_id, records)
        elif format.lower() == "parquet":
            path = self._write_parquet(trace_id, records)
        else:
            logger.warning(f"Unknown format '{format}', defaulting to CSV")
            path = self._write_csv(trace_id, records)
            
        # Ensure all required metadata fields are present
        enhanced_metadata = dict(metadata)
        
        # Add mode/tracing_mode if missing
        if "mode" in metadata and "tracing_mode" not in metadata:
            enhanced_metadata["tracing_mode"] = metadata["mode"]
        elif "tracing_mode" in metadata and "mode" not in metadata:
            enhanced_metadata["mode"] = metadata["tracing_mode"]
        
        # Write both trace-specific and unified metadata for backward compatibility
        self._write_metadata(trace_id, enhanced_metadata)
        self._write_unified_metadata(enhanced_metadata)
        
        return path
        
    def _write_csv(self, trace_id: str, records: List[Dict[str, Any]]) -> str:
        """
        Write trace records to a CSV file with sanitization.
        Uses the same naming pattern as the old EnhancedSemanticTracer.
        
        Args:
            trace_id: Unique identifier for the trace
            records: List of trace record dictionaries
            
        Returns:
            Path to the created CSV file
        """
        if not records:
            return ""
            
        # Sanitize text fields
        sanitized_records = []
        for record in records:
            sanitized_record = {}
            for key, value in record.items():
                if isinstance(value, str):
                    sanitized_record[key] = self.sanitize_for_csv(value)
                elif isinstance(value, float):
                    # Format floating-point numbers with consistent precision
                    sanitized_record[key] = self._format_float_precision(value)
                else:
                    sanitized_record[key] = value
            sanitized_records.append(sanitized_record)
            
        # Create DataFrame from sanitized records
        df = pd.DataFrame(sanitized_records)
        
        # Get the mode from records if available
        mode = records[0].get("mode", "unknown") if records else "unknown"
        
        # Check if trace_id already contains the mode to avoid duplication
        if trace_id.startswith(f"{mode}_"):
            # Already has mode prefix, use as is
            csv_path = os.path.join(self.csv_dir, f"trace_{trace_id}_data.csv")
        else:
            # Add mode prefix to match old format: trace_mode_id_data.csv
            csv_path = os.path.join(self.csv_dir, f"trace_{mode}_{trace_id}_data.csv")
        
        # Write CSV with double quoting for proper escaping
        df.to_csv(csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        
        logger.info(f"Wrote {len(sanitized_records)} records to {csv_path}")
        return csv_path
        
    def _write_parquet(self, trace_id: str, records: List[Dict[str, Any]]) -> str:
        """
        Write trace records to a Parquet file.
        
        Args:
            trace_id: Unique identifier for the trace
            records: List of trace record dictionaries
            
        Returns:
            Path to the created Parquet file
        """
        if not records:
            return ""
            
        # Create DataFrame from records
        df = pd.DataFrame(records)
        
        # Create file path
        parquet_path = os.path.join(self.parquet_dir, f"trace_{trace_id}.parquet")
        
        # Write Parquet
        df.to_parquet(parquet_path, index=False)
        
        logger.info(f"Wrote {len(records)} records to {parquet_path}")
        return parquet_path
        
    def _write_metadata(self, trace_id: str, metadata: Dict[str, Any]) -> str:
        """
        Write trace metadata to a JSON file.
        
        Args:
            trace_id: Unique identifier for the trace
            metadata: Dictionary of trace metadata
            
        Returns:
            Path to the created JSON file
        """
        # Create file path
        json_path = os.path.join(self.meta_dir, f"metadata_{trace_id}.json")
        
        # Convert non-serializable objects
        meta_copy = self._prepare_metadata_for_serialization(metadata)
                
        # Write JSON
        with open(json_path, 'w') as f:
            json.dump(meta_copy, f, indent=2)
            
        logger.info(f"Wrote metadata to {json_path}")
        return json_path
    
    def _write_unified_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Write a unified metadata file for backward compatibility with old version.
        
        Args:
            metadata: Dictionary of trace metadata
            
        Returns:
            Path to the created JSON file
        """
        # Create unified metadata path matching old version
        json_path = os.path.join(self.csv_dir, "trace_metadata.json")
        
        # Check if file exists, update if it does
        existing_meta = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    existing_meta = json.load(f)
            except:
                pass
        
        # Update with new metadata
        # Deep merge for nested metadata (like feature_mapping, target_tokens)
        self._deep_merge_metadata(existing_meta, metadata)
        
        # Ensure all required fields from old version are present
        required_fields = {
            "tracing_mode": metadata.get("mode", metadata.get("tracing_mode", "saliency")),
            "image_available": metadata.get("image_available", False),
            "logit_lens_concepts": metadata.get("logit_lens_concepts", []),
        }
        
        # Add required fields if missing
        for field, default_value in required_fields.items():
            if field not in existing_meta:
                existing_meta[field] = default_value
        
        # Convert non-serializable objects 
        meta_copy = self._prepare_metadata_for_serialization(existing_meta)
                
        # Write JSON
        with open(json_path, 'w') as f:
            json.dump(meta_copy, f, indent=2)
            
        logger.info(f"Wrote unified metadata to {json_path}")
        return json_path
    
    def _deep_merge_metadata(self, target: Dict, source: Dict) -> None:
        """
        Deep merge source into target, preserving nested structures.
        
        Args:
            target: Target dictionary to be updated
            source: Source dictionary with new values
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively merge dictionaries
                self._deep_merge_metadata(target[key], value)
            else:
                # Replace or add value
                target[key] = value
    
    def _prepare_metadata_for_serialization(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert non-serializable objects in metadata to serializable forms.
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Serializable metadata dictionary
        """
        meta_copy = {}
        
        for k, v in metadata.items():
            if isinstance(v, torch.Tensor):
                meta_copy[k] = v.tolist()
            elif isinstance(v, dict):
                meta_copy[k] = self._prepare_metadata_for_serialization(v)
            elif isinstance(v, (int, float, str, bool, list, tuple)) or v is None:
                meta_copy[k] = v
            else:
                meta_copy[k] = str(v)
                
        return meta_copy
        
    @staticmethod
    def load_csv(csv_path: str, dtype: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Load trace records from a CSV file with appropriate dtype handling.
        
        Args:
            csv_path: Path to the CSV file
            dtype: Optional dictionary mapping column names to data types
            
        Returns:
            DataFrame containing trace records
        """
        # Default is None - pandas will infer dtypes automatically
        # High precision columns that should be loaded as float64
        high_precision_cols = ['raw_score', 'll_top1_prob', 'll_top2_prob', 
                              'predicted_top_prob', 'weight']
        
        # Read CSV without forcing dtype conversion
        df = pd.read_csv(csv_path, dtype=dtype)
        
        # After loading, convert specific object columns to appropriate types
        # Only process string columns that might need special handling
        if dtype is None:
            for col in df.select_dtypes(include=['object']).columns:
                if col in high_precision_cols:
                    # Convert high precision columns to float explicitly
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass
        
        return df
    
    @staticmethod
    def load_parquet(parquet_path: str) -> pd.DataFrame:
        """
        Load trace records from a Parquet file.
        
        Args:
            parquet_path: Path to the Parquet file
            
        Returns:
            DataFrame containing trace records
        """
        return pd.read_parquet(parquet_path)
        
    @staticmethod
    def load_metadata(json_path: str) -> Dict[str, Any]:
        """
        Load trace metadata from a JSON file.
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            Dictionary containing trace metadata
        """
        with open(json_path, 'r') as f:
            return json.load(f)
            
    @staticmethod
    def sanitize_for_csv(text: str) -> str:
        """
        Sanitize text for safe CSV storage.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
            
        # Replace problematic characters
        sanitized = text.replace('\n', '\\n')
        sanitized = sanitized.replace('\r', '\\r')
        sanitized = sanitized.replace('\t', '\\t')
        sanitized = sanitized.replace('"', '""')  # Double quotes for CSV escape
        
        return sanitized
    
    @staticmethod
    def _format_float_precision(value: float, precision: int = 6) -> float:
        """
        Format floating-point numbers with consistent precision.
        
        Args:
            value: Float value to format
            precision: Number of decimal places
            
        Returns:
            Formatted float value
        """
        if isinstance(value, float):
            return round(value, precision)
        return value