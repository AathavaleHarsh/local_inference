import subprocess
import time
import threading
import queue
import sys
import tiktoken
from colorama import Fore, Style, init
# import psutil  # Optional for resource monitoring

# Initialize colorama
init(autoreset=True)

def count_tokens(text, encoding):
    """
    Counts the number of tokens in the given text using the specified encoding.
    """
    tokens = encoding.encode(text)
    return len(tokens)

def enqueue_output(pipe, q):
    """
    Reads lines from the subprocess pipe and puts them into a queue.
    """
    for line in iter(pipe.readline, ''):
        q.put(line)
    pipe.close()

# def monitor_resources(interval=1):
#     """
#     Monitors CPU and memory usage at specified intervals.
#     """
#     try:
#         while True:
#             cpu = psutil.cpu_percent(interval=interval)
#             memory = psutil.virtual_memory().percent
#             print(f"{Fore.GREEN}CPU Usage: {cpu}% | Memory Usage: {memory}%")
#     except KeyboardInterrupt:
#         print(f"\n{Fore.YELLOW}Resource monitoring stopped.")

def run_ollama_command(command, encoding):
    """
    Runs the given Ollama command, counts tokens in real-time, and measures inference speed.
    """
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,      # Redirect stderr to stdout
            shell=True,
            bufsize=1,
            text=True,                     # Equivalent to universal_newlines=True
            encoding='utf-8',              # Explicitly set encoding to 'utf-8'
            errors='replace',              # Replace undecodable bytes with a placeholder
            stdin=subprocess.DEVNULL       # Prevent subprocess from reading stdin
        )
    except Exception as e:
        print(f"{Fore.RED}Failed to start the Ollama process: {e}")
        sys.exit(1)

    q = queue.Queue()
    t = threading.Thread(target=enqueue_output, args=(process.stdout, q))
    t.daemon = True  # Ensures thread exits when main program does
    t.start()

    # Start resource monitoring in a separate thread (optional)
    # resource_thread = threading.Thread(target=monitor_resources, args=(1,), daemon=True)
    # resource_thread.start()

    total_tokens = 0
    start_time = time.time()

    print(f"{Fore.GREEN}Starting inference...")

    try:
        while True:
            try:
                line = q.get_nowait()
            except queue.Empty:
                if process.poll() is not None:
                    break
                time.sleep(0.1)
                continue
            if line:
                # Optionally, print the output with coloring
                print(f"{Fore.CYAN}{line}", end='')  # Remove Fore.CYAN if color is not desired
                tokens = count_tokens(line, encoding)
                total_tokens += tokens
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Process interrupted by user.")
        process.terminate()
        sys.exit(1)

    end_time = time.time()
    inference_time = end_time - start_time
    tokens_per_second = total_tokens / inference_time if inference_time > 0 else float('inf')

    print(f"\n{Fore.BLUE}=== Inference Summary ===")
    print(f"{Fore.BLUE}Total Tokens: {Fore.YELLOW}{total_tokens}")
    print(f"{Fore.BLUE}Inference Time: {Fore.YELLOW}{inference_time:.2f} seconds")
    print(f"{Fore.BLUE}Inference Speed: {Fore.YELLOW}{tokens_per_second:.2f} tokens/second")

    # Optionally, log the results
    log_results(total_tokens, inference_time, tokens_per_second)

def log_results(total_tokens, inference_time, tokens_per_second, logfile="inference_log.txt"):
    """
    Logs the inference results to a file.
    """
    try:
        with open(logfile, "a", encoding='utf-8') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}, {total_tokens}, {inference_time:.2f}, {tokens_per_second:.2f}\n")
        print(f"{Fore.MAGENTA}Results logged to {logfile}")
    except Exception as e:
        print(f"{Fore.RED}Failed to log results: {e}")

def main():
    """
    Main function to execute the inference speed measurement.
    """
    # User Inputs
    model = input("Enter the Ollama model name (e.g., 'llama2'): ").strip()
    prompt = input("Enter the prompt for the model: ").strip()

    # Construct the Ollama command with the prompt as a positional argument
    # Ensure that the prompt is enclosed in quotes to handle multi-word prompts
    ollama_command = f'ollama run {model} "{prompt}"'

    # Initialize the tokenizer encoding
    # Replace 'gpt2' with your model's encoding if different
    try:
        encoding = tiktoken.get_encoding("gpt2")
    except Exception as e:
        print(f"{Fore.RED}Error initializing tokenizer: {e}")
        sys.exit(1)

    # Run the Ollama command and measure inference speed
    run_ollama_command(ollama_command, encoding)

if __name__ == "__main__":
    main()
