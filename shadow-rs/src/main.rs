use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use anyhow::{Result, Context};
use clap::Parser;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "../mvc/model.safetensors")]
    weight_path: String,
    
    #[arg(short, long, default_value = "../mvc/dashboard/neuron_metadata.json")]
    output_path: String,
}

#[derive(Serialize, Deserialize)]
struct NeuronMetadata {
    id: usize,
    role: String,
    norm: f32,
    sparsity: f32,
    semantic: String,
}

#[derive(Serialize, Deserialize)]
struct LayerData {
    attn: Vec<NeuronMetadata>,
    mlp: Vec<NeuronMetadata>,
}

struct RustProfiler {
    device: Device,
}

impl RustProfiler {
    fn new() -> Result<Self> {
        let device = Device::Cpu;
        Ok(Self { device })
    }

    fn classify_role(norm: f32, sparsity: f32) -> String {
        // Calibrated to GPT2-Small statistics (Mean Norm ~2.6, Min ~1.9)
        if norm > 3.2 { "Scaler (Amplifier)".to_string() }
        else if norm < 2.3 { "Inhibitor (Noise Gate)".to_string() }
        else if sparsity < 0.20 { "Projector (Feature Detector)".to_string() }
        else if sparsity > 0.35 { "Composer (Global Linker)".to_string() }
        else { "Rotator (Context Tracker)".to_string() }
    }

    fn mock_semantic(layer: usize, n_type: &str, idx: usize) -> String {
        let meanings = [
            "Detects definite articles", "Tracks plural nouns", "Inhibits noise in context",
            "Anticipates verb phrases", "Links subjects to objects", "Encodes spatial relations",
            "Triggers factual recall", "Filters for animal names", "Maintains dialogue state",
            "Detects punctuation cues", "Amplifies important semantic tokens", "Routes dependency info"
        ];
        // We use a slightly different seed for reproducibility in the demo
        meanings[(layer + idx + if n_type == "mlp" { 3 } else { 0 }) % meanings.len()].to_string()
    }

    fn profile_layer(&self, vb: &VarBuilder, layer_idx: usize) -> Result<LayerData> {
        let mut layer_data = LayerData { attn: vec![], mlp: vec![] };

        // 1. Attention Output Projection
        let attn_key = format!("transformer.h.{}.attn.c_proj.weight", layer_idx);
        let w_attn = vb.get((768, 768), &attn_key)?; // Transposed in GPT2 (768, 768)
        
        let norm_attn = w_attn.sqr()?.sum(1)?.sqrt()?;
        let abs_attn = w_attn.abs()?;
        let threshold = Tensor::new(0.1f32, &self.device)?;
        let mask_attn = abs_attn.ge(&threshold.broadcast_as(abs_attn.shape())?)?;
        let sparsity_attn = mask_attn.to_dtype(candle_core::DType::F32)?.sum(1)?
            .affine(1.0 / 768.0, 0.0)?;

        let norms = norm_attn.to_vec1::<f32>()?;
        let sparsities = sparsity_attn.to_vec1::<f32>()?;

        for n in 0..768 {
            layer_data.attn.push(NeuronMetadata {
                id: n,
                role: Self::classify_role(norms[n], sparsities[n]),
                norm: (norms[n] * 1000.0).round() / 1000.0,
                sparsity: (sparsities[n] * 1000.0).round() / 1000.0,
                semantic: Self::mock_semantic(layer_idx, "attn", n),
            });
        }

        // 2. MLP Output Projection
        let mlp_key = format!("transformer.h.{}.mlp.c_proj.weight", layer_idx);
        let w_mlp = vb.get((3072, 768), &mlp_key)?; // (3072, 768)
        
        let norm_mlp = w_mlp.sqr()?.sum(1)?.sqrt()?;
        let abs_mlp = w_mlp.abs()?;
        let mask_mlp = abs_mlp.ge(&threshold.broadcast_as(abs_mlp.shape())?)?;
        let sparsity_mlp = mask_mlp.to_dtype(candle_core::DType::F32)?.sum(1)?
            .affine(1.0 / 768.0, 0.0)?;

        let norms_mlp = norm_mlp.to_vec1::<f32>()?;
        let sparsities_mlp = sparsity_mlp.to_vec1::<f32>()?;

        for n in 0..3072 {
            layer_data.mlp.push(NeuronMetadata {
                id: n,
                role: Self::classify_role(norms_mlp[n], sparsities_mlp[n]),
                norm: (norms_mlp[n] * 1000.0).round() / 1000.0,
                sparsity: (sparsities_mlp[n] * 1000.0).round() / 1000.0,
                semantic: Self::mock_semantic(layer_idx, "mlp", n),
            });
        }

        Ok(layer_data)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("Initializing RUST FULL-SCALE PROFILER...");
    println!("Loading weights from: {}", args.weight_path);
    
    let profiler = RustProfiler::new()?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[args.weight_path], candle_core::DType::F32, &profiler.device)? };
    
    let mut metadata = HashMap::new();
    let start = std::time::Instant::now();

    for layer_idx in 0..12 {
        print!("  Profiling Layer {}/12... ", layer_idx + 1);
        std::io::stdout().flush()?;
        let layer_start = std::time::Instant::now();
        let data = profiler.profile_layer(&vb, layer_idx)?;
        metadata.insert(format!("layer_{}", layer_idx), data);
        println!("Complete ({:?})", layer_start.elapsed());
    }

    println!("Full Sweep Complete in {:?}", start.elapsed());
    
    println!("Saving metadata to: {}", args.output_path);
    let json = serde_json::to_string(&metadata)?;
    let mut file = File::create(args.output_path)?;
    file.write_all(json.as_bytes())?;
    
    println!("✓ DASHBOARD DATA SYNCHRONIZED VIA RUST ENGINE");
    Ok(())
}
