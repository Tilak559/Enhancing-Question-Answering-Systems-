import timeit
from click import prompt
from langchain import LLMBashChain, LlamaCpp

# Start timer
start = timeit.default_timer()


output = LLMBashChain(prompt,
             max_tokens=-1,
             echo=False,
             temperature=0.1,
             top_p=0.9)

# Stop timer
stop = timeit.default_timer()
duration = stop - start
print("Time: ", duration, '\n\n')

# Display generated text
print(output['choices'][0]['text'])

# Write to file
with open("response.txt", "a") as f:
  f.write(f"Time: {duration}")
  f.write(output['choices'][0]['text'])