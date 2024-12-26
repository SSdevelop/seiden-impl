import torch
from transformers import AutoProcessor, GroundingDinoForObjectDetection


def create_index(model, query, sampled_keyframe_indices, sampled_keyframes):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    model = GroundingDinoForObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base"
    ).to(device)

    index = {}
    for i, frame in enumerate(sampled_keyframes):
        inputs = processor(images=frame, text=query, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            output,
            inputs.input_ids,
            box_threshold=0.3,
            text_threshold=0.2,
            target_sizes=torch.tensor([[frame.shape[0], frame.shape[1]]]),
        )
        index[sampled_keyframe_indices[i]] = result[0]["scores"].cpu()
    return index
