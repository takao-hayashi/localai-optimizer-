import argparse
from localai_optimizer.optimize import optimize_model
from localai_optimizer.quantize import quantize_model
from localai_optimizer.convert import convert_model
from localai_optimizer.benchmark import benchmark_model
from localai_optimizer.report import generate_report

def main():
    parser = argparse.ArgumentParser(description="LocalAI Optimizer CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize a model")
    optimize_parser.add_argument("model", type=str, help="Path or name of the model to optimize")
    optimize_parser.add_argument("--precision", type=str, default="int8", help="Precision for quantization (e.g., int8, int4)")
    optimize_parser.add_argument("--target", type=str, default="gguf", help="Target format for conversion (e.g., gguf, onnx)")
    optimize_parser.set_defaults(func=optimize_model)

    # Quantize command
    quantize_parser = subparsers.add_parser("quantize", help="Quantize a model")
    quantize_parser.add_argument("model", type=str, help="Path or name of the model to quantize")
    quantize_parser.add_argument("--precision", type=str, default="int8", help="Precision for quantization (e.g., int8, int4)")
    quantize_parser.set_defaults(func=quantize_model)

    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert a model")
    convert_parser.add_argument("model", type=str, help="Path or name of the model to convert")
    convert_parser.add_argument("--target", type=str, default="onnx", help="Target format for conversion (e.g., onnx, torchscript)")
    convert_parser.set_defaults(func=convert_model)

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark a model")
    benchmark_parser.add_argument("model", type=str, help="Path to the optimized model directory")
    benchmark_parser.set_defaults(func=benchmark_model)

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate a report for a model")
    report_parser.add_argument("model", type=str, help="Path to the optimized model directory")
    report_parser.set_defaults(func=generate_report)

    args = parser.parse_args()

    if hasattr(args, "func"):
        if args.command == "optimize":
            args.func(args.model, args.precision, args.target)
        elif args.command == "quantize":
            args.func(args.model, args.precision)
        elif args.command == "convert":
            args.func(args.model, args.target)
        elif args.command == "benchmark" or args.command == "report":
            args.func(args.model)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()


