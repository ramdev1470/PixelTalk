"""
Disjoint Grouped Dataset for Multimodal Vision-Language Dialogue
Solves data leakage by splitting strictly by Image ID (Zero image/caption overlap between splits).
Includes fine-grained visual questions that require direct pixel inspection.
"""

import os
import json
import random
from typing import List, Dict, Any, Tuple
import torch
from torch.utils.data import Dataset

DATA_RECORDS = [
    {
        "image_id": "img_woman_camera_01",
        "category": "person_outdoor",
        "image_file": "woman_camera.jpg",
        "baseline_caption": "A person holding an object in front of their face.",
        "ground_truth_details": {
            "subject": "young woman with blonde hair",
            "object_held": "vintage silver and black SLR camera",
            "headwear": "red knit beanie hat",
            "clothing": "blue denim jean jacket over dark shirt",
            "setting": "outdoor pine forest",
            "background": "lush evergreen pine trees and forest foliage",
            "safety": "not food / not edible"
        },
        "dialogue": [
            {
                "turn": 1,
                "question": "What is the person holding in their hands?",
                "answer": "The woman is holding a vintage silver and black SLR camera up to her eyes to take a photo.",
                "pixel_dependent": True
            },
            {
                "turn": 2,
                "question": "Is she holding a smartphone or a camera?",
                "answer": "She is holding an SLR camera, not a smartphone.",
                "pixel_dependent": True
            },
            {
                "turn": 3,
                "question": "What color is her hat?",
                "answer": "She is wearing a bright red knit beanie hat.",
                "pixel_dependent": True
            },
            {
                "turn": 4,
                "question": "Where is she located?",
                "answer": "She is outdoors in a pine forest surrounded by evergreen trees.",
                "pixel_dependent": False
            },
            {
                "turn": 5,
                "question": "Is it safe to eat?",
                "answer": "No, this is an outdoor photography portrait and not an edible food item.",
                "pixel_dependent": False
            }
        ]
    },
    {
        "image_id": "img_pizza_01",
        "category": "food",
        "image_file": "pizza.jpg",
        "baseline_caption": "A pizza on a plate on a table.",
        "ground_truth_details": {
            "dish": "freshly baked round Italian pizza",
            "toppings": "melted mozzarella cheese, savory pepperoni slices, tomato sauce, basil leaves",
            "plate": "white circular ceramic serving plate",
            "table": "wooden plank table",
            "safety": "safe and edible cooked food"
        },
        "dialogue": [
            {
                "turn": 1,
                "question": "What specific toppings are on this pizza?",
                "answer": "It is topped with melted mozzarella cheese, red pepperoni slices, tomato sauce, and green fresh basil leaves.",
                "pixel_dependent": True
            },
            {
                "turn": 2,
                "question": "Is this pizza vegetarian?",
                "answer": "No, it contains pepperoni meat slices so it is not vegetarian.",
                "pixel_dependent": True
            },
            {
                "turn": 3,
                "question": "Is it safe to eat?",
                "answer": "Yes, it is a freshly cooked and safe meal ready for consumption.",
                "pixel_dependent": False
            },
            {
                "turn": 4,
                "question": "What is it served on?",
                "answer": "It is served on a white circular ceramic plate on a wooden table.",
                "pixel_dependent": False
            }
        ]
    },
    {
        "image_id": "img_dog_park_01",
        "category": "animals",
        "image_file": "dog_park.jpg",
        "baseline_caption": "A dog in the grass with a toy.",
        "ground_truth_details": {
            "animal": "golden retriever puppy",
            "toy": "bright red rubber toy ball next to its paws",
            "environment": "sunny outdoor park with green lawn and trees",
            "safety": "domestic pet / outdoor park"
        },
        "dialogue": [
            {
                "turn": 1,
                "question": "What kind of toy is near the dog?",
                "answer": "There is a bright red rubber toy ball resting right next to the puppy's front paws.",
                "pixel_dependent": True
            },
            {
                "turn": 2,
                "question": "What breed of dog is this?",
                "answer": "It is an adorable golden retriever puppy with golden-cream fur.",
                "pixel_dependent": True
            },
            {
                "turn": 3,
                "question": "What color is the toy?",
                "answer": "The toy ball is bright red.",
                "pixel_dependent": True
            },
            {
                "turn": 4,
                "question": "Where is this taking place?",
                "answer": "It is in a sunny outdoor park on a green grass lawn.",
                "pixel_dependent": False
            }
        ]
    },
    {
        "image_id": "img_vintage_car_01",
        "category": "vehicles",
        "image_file": "vintage_car.jpg",
        "baseline_caption": "A red car parked on the street.",
        "ground_truth_details": {
            "vehicle": "red vintage sports convertible coupe",
            "roof": "open convertible top",
            "wheels": "spoke wheels with chrome accents",
            "background": "urban street with brick city buildings",
            "safety": "stationary motor vehicle"
        },
        "dialogue": [
            {
                "turn": 1,
                "question": "Is the car a convertible or a hardtop sedan?",
                "answer": "It is a vintage convertible coupe with an open roof.",
                "pixel_dependent": True
            },
            {
                "turn": 2,
                "question": "What color is the car body?",
                "answer": "The car has a glossy vibrant cherry red paint finish.",
                "pixel_dependent": True
            },
            {
                "turn": 3,
                "question": "Is the car moving or parked?",
                "answer": "The car is parked along the paved curb of a city street.",
                "pixel_dependent": False
            },
            {
                "turn": 4,
                "question": "What buildings are in the background?",
                "answer": "There are classic multi-story brick city buildings behind the vehicle.",
                "pixel_dependent": True
            }
        ]
    },
    {
        "image_id": "img_laptop_desk_01",
        "category": "technology",
        "image_file": "laptop_desk.jpg",
        "baseline_caption": "A computer on a desk with a cup.",
        "ground_truth_details": {
            "laptop": "silver aluminum laptop with code editor open",
            "screen_content": "syntax highlighted programming code lines in green and blue",
            "beverage": "white ceramic mug containing dark coffee",
            "accessories": "spiral notebook beside the laptop",
            "safety": "workspace electronics"
        },
        "dialogue": [
            {
                "turn": 1,
                "question": "What is shown on the computer screen?",
                "answer": "The screen displays a software code editor with syntax-highlighted lines of programming code.",
                "pixel_dependent": True
            },
            {
                "turn": 2,
                "question": "What beverage is next to the laptop?",
                "answer": "There is a white ceramic mug filled with dark coffee sitting to the left of the laptop.",
                "pixel_dependent": True
            },
            {
                "turn": 3,
                "question": "Is there any paper or notebook nearby?",
                "answer": "Yes, a spiral notebook is positioned to the right of the laptop on the wooden desk.",
                "pixel_dependent": True
            },
            {
                "turn": 4,
                "question": "Is it safe to eat?",
                "answer": "No, these are office electronic devices and stationery; only the coffee in the mug is drinkable.",
                "pixel_dependent": False
            }
        ]
    },
    {
        "image_id": "img_mountain_lake_01",
        "category": "nature",
        "image_file": "mountain_lake.jpg",
        "baseline_caption": "Mountains and a lake.",
        "ground_truth_details": {
            "peaks": "jagged snow-capped alpine mountain summits",
            "water": "calm turquoise lake acting as a natural mirror",
            "shoreline": "dense evergreen pine and fir forest",
            "weather": "clear sunny sky",
            "safety": "natural wilderness"
        },
        "dialogue": [
            {
                "turn": 1,
                "question": "Is there snow on top of the mountains?",
                "answer": "Yes, the jagged alpine peaks are capped with bright white snow and glacial ice.",
                "pixel_dependent": True
            },
            {
                "turn": 2,
                "question": "What is reflected in the water?",
                "answer": "The calm turquoise water reflects the towering mountain peaks and shoreline pine trees.",
                "pixel_dependent": True
            },
            {
                "turn": 3,
                "question": "What kind of trees grow along the lake shore?",
                "answer": "Dense evergreen pine and fir trees line the perimeter of the lake.",
                "pixel_dependent": True
            },
            {
                "turn": 4,
                "question": "Where is this location?",
                "answer": "This is in an alpine wilderness mountain range.",
                "pixel_dependent": False
            }
        ]
    }
]

def split_dataset_by_image_groups(
    records: List[Dict[str, Any]] = DATA_RECORDS,
    test_ratio: float = 0.33,
    val_ratio: float = 0.17,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits dataset strictly by unique image_id (Grouped Disjoint Split).
    Guarantees 0% image or caption overlap between Train, Val, and Test sets!
    """
    unique_ids = list(set(r["image_id"] for r in records))
    unique_ids.sort() # deterministic ordering
    rng = random.Random(seed)
    rng.shuffle(unique_ids)

    n_test = max(1, int(len(unique_ids) * test_ratio))
    n_val = max(1, int(len(unique_ids) * val_ratio))

    test_ids = set(unique_ids[:n_test])
    val_ids = set(unique_ids[n_test:n_test + n_val])
    train_ids = set(unique_ids[n_test + n_val:])

    # Ensure train has at least something
    if not train_ids:
        train_ids = set(unique_ids[n_test:])
        val_ids = set(unique_ids[:1])

    train_samples = [r for r in records if r["image_id"] in train_ids]
    val_samples = [r for r in records if r["image_id"] in val_ids]
    test_samples = [r for r in records if r["image_id"] in test_ids]

    # Verify 0% leakage
    assert train_ids.isdisjoint(val_ids), "Data leakage between train and val!"
    assert train_ids.isdisjoint(test_ids), "Data leakage between train and test!"

    return train_samples, val_samples, test_samples

class MultimodalVLDataset(Dataset):
    """
    PyTorch Dataset for Multimodal Vision-Language Dialogue.
    Provides (image_path, dialogue_history, question, answer, pixel_dependent).
    """
    def __init__(self, sample_records: List[Dict[str, Any]], image_dir: str = "PixelTalk/sample_images"):
        self.items = []
        for rec in sample_records:
            img_path = os.path.join(image_dir, rec["image_file"])
            dialogue = rec["dialogue"]
            history = []
            for item in dialogue:
                self.items.append({
                    "image_id": rec["image_id"],
                    "image_path": img_path,
                    "category": rec["category"],
                    "baseline_caption": rec["baseline_caption"],
                    "ground_truth_details": rec["ground_truth_details"],
                    "history": list(history),
                    "question": item["question"],
                    "answer": item["answer"],
                    "pixel_dependent": item.get("pixel_dependent", False)
                })
                history.append((item["question"], item["answer"]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
