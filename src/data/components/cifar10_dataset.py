import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer, CLIPImageProcessor

_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class Cifar10Dataset(Dataset):
    """CIFAR-10 wrapped as an image-captioning dataset for Stage-1 VLM pretraining.

    Each image's label is converted to a short caption: "a photo of a {class}".
    For instruction-tuned LMs (e.g. SmolLM2-Instruct), the text is formatted with
    the model's chat template:
        <|im_start|>user
        {user_prompt}<|im_end|>
        <|im_start|>assistant
        a photo of a {class}<|im_end|>
    Loss is computed only on the assistant's response (everything before it is
    masked with -100). At generation time the callback feeds the same user
    prompt + assistant header so the LM sees the in-distribution format.
    """

    def __init__(
        self,
        split: str = "train",
        vision_model_name: str = "openai/clip-vit-base-patch32",
        lm_model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        max_length: int = 48,
        max_samples: int | None = None,
        user_prompt: str = "Describe this image briefly.",
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.user_prompt = user_prompt

        self.image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.hf_dataset = load_dataset("uoft-cs/cifar10", split=split)
        if max_samples is not None:
            self.hf_dataset = self.hf_dataset.select(range(min(max_samples, len(self.hf_dataset))))

        # Dataset interface consumed by CaptionSamplerCallback
        self.image_key = "img"
        self.items_per_image = 1

    def get_reference(self, row: dict) -> list[str]:
        return [f"a photo of a {_CLASSES[row['label']]}"]

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.hf_dataset[idx]

        pixel_values = self.image_processor(images=row["img"].convert("RGB"), return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)  # [3, 224, 224]

        caption = f"a photo of a {_CLASSES[row['label']]}"
        messages = [
            {"role": "user", "content": self.user_prompt},
            {"role": "assistant", "content": caption},
        ]

        # Full conversation = user + assistant; used for both input_ids and labels
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        full_ids = torch.tensor(
            self.tokenizer.encode(full_text, add_special_tokens=False),
            dtype=torch.long,
        )

        # Prompt-only = user + assistant header; everything before this index is masked
        prompt_text = self.tokenizer.apply_chat_template(
            messages[:1],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_len = len(self.tokenizer.encode(prompt_text, add_special_tokens=False))

        # Truncate to max_length, then pad
        full_ids = full_ids[: self.max_length]
        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full((self.max_length,), pad_id, dtype=torch.long)
        input_ids[: len(full_ids)] = full_ids

        # Length-based mask (pad_token == eos_token for SmolLM2, so can't compare ids)
        attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        attention_mask[: len(full_ids)] = 1

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # mask user prompt
        labels[attention_mask == 0] = -100  # mask padding

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
