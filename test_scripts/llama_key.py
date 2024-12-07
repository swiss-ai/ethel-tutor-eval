import openai

# swissai/ethel-70b-magpie
# swissai/ethel-70b-pretrain
# meta-llama/Meta-Llama-3.1-70B-Instruct
# swissai/ethel-70b-tutorchat


client = openai.Client(api_key="sk-rc-XcyV-GtgecNFEEzOJRyxiw", base_url="https://fmapi.swissai.cscs.ch")
res = client.chat.completions.create(
    model="swissai/ethel-70b-tutorchat",
    messages=[
        {
            "content": "Who is Pablo Picasso?", 
            "role": "user",
        }
    ],
    stream=True,
)

for chunk in res:
    if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
