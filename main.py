import argparse

from index_contruction import create_index
from input_processor import extract_keyframes, sample_keyframes


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Seiden Script. Enter video path, text query, huggingface model path"
    )

    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to the input video"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="The object you want to find in the video",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="IDEA-Research/grounding-dino-base",
        help="HuggingFace model path (default: grounding-dino-base)",
    )
    parser.add_argument(
        "--sampling_ratio", type=float, default=1, help="Sampling Ratio for Seiden"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # 0. Parse input video path, text query and model name
    args = parse_arguments()
    # 1. Extract Keyframes from the video
    print("Extracting Keyframes...")
    keyframe_indices, keyframes = extract_keyframes(args.video_path)
    print(f"Extracted {len(keyframe_indices)} keyframes.")
    # 2. Sample keyframes
    print("Sampling Keyframes...")
    sampled_keyframe_indices, sampled_keyframes = sample_keyframes(
        keyframe_indices, keyframes, args.sampling_ratio
    )
    print(f"Sampled {len(sampled_keyframe_indices)} keyframes.")
    # 3. Create index
    print("Creating index...")
    index = create_index(
        args.model, args.query, sampled_keyframe_indices, sampled_keyframes
    )
    print("Created index on sampled keyframes")
    # 4. MAB Sampling
    # 5. Label Propogation
    # 6. SUPG
    # 7. Output
    print(index)
