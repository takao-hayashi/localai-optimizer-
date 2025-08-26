# LocalAI Optimizer ⚡

**One-command optimization tool for AI models — quantize, minimize, and convert for local inference.**

![Demo GIF placeholder](/demo.gif)

## Features

✅ **Quantize**: Reduce model size and accelerate inference with INT8/INT4 quantization (GGUF/ONNX).
✅ **Convert**: Export models to various formats like ONNX, TorchScript, CoreML, and TF-Lite.
✅ **Benchmark**: Measure inference latency and memory usage for optimized models.
✅ **Report**: Generate comprehensive Markdown and JSON reports with optimization results.
✅ **Broad Model Support**: Works with Hugging Face Transformers (Llama, Mistral, Phi, Qwen, etc.) and local `.bin/.safetensors` files.
✅ **CLI**: Easy-to-use command-line interface for seamless optimization workflows.
✅ **Configurable**: Customize precision (FP32, FP16, INT8, INT4) and target devices (CPU, GPU, Mobile).
✅ **User-Friendly**: Features progress bars (`tqdm`), pretty CLI output (`rich`), and dry-run previews.

## Quick Start

📦 **Installation**

```bash
pip install localai-optimizer
```

🚀 **Example CLI Usage**

```bash
# Optimize a Hugging Face model
localai optimize facebook/opt-125m

# Quantize a local model to INT4
localai quantize my_model.bin --precision int4

# Benchmark an optimized model
localai benchmark ./output/facebook-opt-125m/

# Generate a report
localai report ./output/facebook-opt-125m/
```

## Examples / Use Cases

### Llama-2 → INT8 Quantization

Optimize a Llama-2 model for efficient local inference:

```bash
localai optimize meta-llama/Llama-2-7b-hf --precision int8 --target gguf
```

### Export to ONNX

Convert a PyTorch model to ONNX format for cross-platform deployment:

```bash
localai convert distilbert-base-uncased --target onnx
```

## Roadmap

- [ ] Support for more model architectures and frameworks.
- [ ] Advanced quantization techniques (e.g., mixed-precision).
- [ ] Integration with cloud-based optimization services.
- [ ] GUI for easier model management and optimization.

## Contributing

We welcome contributions! Please see our `CONTRIBUTING.md` (coming soon) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

⭐ Star this repo if you find it useful!

