# Fine tuning multimodal step by step

## 1. Setup development environment
First step is to install Hugging Face Libraries and Pyroch, including trl, transformers and datasets.

```python
# Install Pytorch & other libraries
%pip install "torch==2.4.0" tensorboard pillow

# Install Hugging Face libraries
%pip install  --upgrade \
  "transformers==4.45.1" \
  "datasets==3.0.1" \
  "accelerate==0.34.2" \
  "evaluate==0.4.3" \
  "bitsandbytes==0.44.0" \
  "trl==0.11.1" \
  "peft==0.13.0" \
  "qwen-vl-utils"
```
Login the HuggingFace Hub to automatically push our model, logs and information to the Hub during training.
```python
from huggingface_hub import login

login(
  token="", # ADD YOUR TOKEN HERE
  add_to_git_credential=True
)
```

## 2. Create and prepare dataset
TRL supports popular instruction and conversation dataset formats. This means we only need to convert our dataset to one of the supported formats and trl will take care of the rest.

```python
"messages": [
  {"role": "system", "content": [{"type":"text", "text": "You are a helpful...."}]},
  {"role": "user", "content": [{
    "type": "text", "text":  "How many dogs are in the image?", 
    "type": "image", "text": <PIL.Image> 
    }]},
  {"role": "assistant", "content": [{"type":"text", "text": "There are 3 dogs in the image."}]}
],
```

In our example we are going to load our dataset using the Datasets library and apply our frompt and convert it into the the conversational format.

Defining instruction prompt.

```python
# note the image is not provided in the prompt its included as part of the "processor"
prompt= """Create a Short Product description based on the provided ##PRODUCT NAME## and ##CATEGORY## and image. 
Only return description. The description should be SEO optimized and for a better mobile search experience.

##PRODUCT NAME##: {product_name}
##CATEGORY##: {category}"""

system_message = "You are an expert product description writer for Amazon."
```

Example format dataset
```
from datasets import load_dataset

# Convert dataset to OAI messages       
def format_data(sample):
    return {"messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt.format(product_name=sample["Product Name"], category=sample["Category"]),
                        },{
                            "type": "image",
                            "image": sample["image"],
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["description"]}],
                },
            ],
        }

# Load dataset from the hub
dataset_id = "philschmid/amazon-product-descriptions-vlm"
dataset = load_dataset("philschmid/amazon-product-descriptions-vlm", split="train")

# Convert dataset to OAI messages
# need to use list comprehension to keep Pil.Image type, .mape convert image to bytes
dataset = [format_data(sample) for sample in dataset]

print(dataset[345]["messages"])
```

## 3. Fine-tune VLM using trl and the SFTTrainer
We will use the SFTTrainer from trl to fine-tune our model. The SFTTrainer makes it straightfoward to supervise fine-tune open LLMs and VLMs. The SFTTrainer is a subclass of the Trainer from the transformers library and supports all the same features, including logging, evaluation, and checkpointing, but adds additiional quality of life features.

We will use the PEFT features in our example. As peft method we will use QLoRA a technique to reduce the memory footprint of large language models during finetuning, without sacrificing performance by using quantization.
Note: We cannot use Flash Attention as we need to pad our multimodal inputs.

We are going to use Qwen 2 VL 7B model, but we can easily swap out the model for another model, including Meta AI's Llama-3.2-11B-Vision, Mistral AI's Pixtral-12B or any other LLMs by changing our model_id variable. We will use bitsandbytes to quantize our model to 4-bit.

Note: Be aware the bigger the model the more memory it will require. In our example we will use the 7B version, which can be tuned on 24GB GPUs.

Correctly, preparing the LLM, Tokenizer and Processor for training VLMs is crucial. The Processor is responsible for including the special tokens and image features in the input.

```python
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

# Hugging Face model id
model_id = "Qwen/Qwen2-VL-7B-Instruct" 

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    device_map="auto",
    # attn_implementation="flash_attention_2", # not supported for training
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
processor = AutoProcessor.from_pretrained(model_id)
```

```python
# Preparation for inference
text = processor.apply_chat_template(
    dataset[2]["messages"], tokenize=False, add_generation_prompt=False
)
print(text)
```

The SFTTrainer  supports a native integration with peft, which makes it super easy to efficiently tune LLMs using, e.g. QLoRA. We only need to create our LoraConfig and provide it to the trainer. Our LoraConfig parameters are defined based on the qlora paper and sebastian's blog post.

```python
from peft import LoraConfig

# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM", 
)
```

Before we can start our training we need to define the hyperparameters (SFTConfig) we want to use and make sure our inputs are correcty provided to the model. Different to text-only supervised fine-tuning we need to provide the image to the model as well. Therefore we create a custom DataCollator which formates the inputs correctly and include the image features. We use the process_vision_info method from a utility package the Qwen2 team provides. If you are using another model, e.g. Llama 3.2 Vision you might have to check if that creates the same processsed image information.

```python
from trl import SFTConfig
from transformers import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

args = SFTConfig(
    output_dir="qwen2-7b-instruct-amazon-description", # directory to save and repository id
    num_train_epochs=3,                     # number of training epochs
    per_device_train_batch_size=4,          # batch size per device during training
    gradient_accumulation_steps=8,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=5,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=True,                       # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
    gradient_checkpointing_kwargs = {"use_reentrant": False}, # use reentrant checkpointing
    dataset_text_field="", # need a dummy field for collator
    dataset_kwargs = {"skip_prepare_dataset": True} # important for collator
)
args.remove_unused_columns=False

# Create a data collator to encode text and image pairs
def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    image_inputs = [process_vision_info(example["messages"])[0] for example in examples]

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100  #
    # Ignore the image token index in the loss computation (model specific)
    if isinstance(processor, Qwen2VLProcessor):
        image_tokens = [151652,151653,151655]
    else: 
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch
```

We now have every building block we need to create our SFTTrainer to start then training our model.

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=collate_fn,
    dataset_text_field="", # needs dummy value
    peft_config=peft_config,
    tokenizer=processor.tokenizer,
)
```

Start training our model by calling the train() method on our Trainer instance. This will start the training loop and train our model for 3 epochs. Since we are using a PEFT method, we will only save the adapted model weights and not the full model.

```python
# start training, the model will be automatically saved to the hub and the output directory
trainer.train()

# save model 
trainer.save_model(args.output_dir)
```

```python
# free the memory again
del model
del trainer
torch.cuda.empty_cache()
```

## 3. Test Model and run Inference
After the training is done we want to evaluate and test our model. First we will load the base model and let it generate a description for a random Amazon product. Then we will load our Q-LoRA adapted model and let it generate a description for the same product.

Finally we can merge the adapter into the base model to make it more efficient and run inference on the same product again.

```python
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

adapter_path = "./qwen2-7b-instruct-amazon-description"

# Load Model base model
model = AutoModelForVision2Seq.from_pretrained(
  model_id,
  device_map="auto",
  torch_dtype=torch.float16
)
processor = AutoProcessor.from_pretrained(model_id)
```