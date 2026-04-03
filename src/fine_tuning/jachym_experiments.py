from transformers import pipeline


def llama_3_2_1b_instruct_pipeline():
    pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct")
    messages = [
        {"role": "user", "content": "34.8*.181+144.2/8974="},
    ]
    return pipe(messages)


if __name__ == '__main__':
    print(llama_3_2_1b_instruct_pipeline())
