from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

def main(src, dest, bits=4, group_size=128):
    tokenizer = AutoTokenizer.from_pretrained(src, use_fast=True)
    example = tokenizer(
        "auto_gptq is a useful tool that can automatically compress model into 4-bit or even higher rate by using GPTQ algorithm.",
        return_tensors="pt", return_token_type_ids=False
    )

    quantize_config = BaseQuantizeConfig(
        bits=bits,  # quantize model
        group_size=group_size,  # it is recommended to set the value to 128
    )

    # load un-quantized model, the model will always be force loaded into cpu
    model = AutoGPTQForCausalLM.from_pretrained(src, quantize_config)

    # quantize model, the examples should be list of dict whose keys contains "input_ids" and "attention_mask"
    # with value under torch.LongTensor type.
    model.quantize([example])

    # save quantized model using safetensors
    model.save_quantized(dest, use_safetensors=True)
    tokenizer.save_pretrained(dest)

if __name__ == "__main__":
    import logging
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    parser.add_argument("--dest", type=str)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    main(args.src, args.dest, args.bits, args.group_size)
