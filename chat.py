#!/usr/bin/env python3

import os
import subprocess
from typing import Optional, List
from llama_models.llama3.api.datatypes import RawMessage
from llama_models.llama3.reference_impl.generation import Llama

def setup_model(
    ckpt_dir: str,
    max_seq_len: int = 1024,
    max_batch_size: int = 1,
    model_parallel_size: Optional[int] = None,
) -> Llama:
    """Initialize the Llama model."""
    return Llama.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=model_parallel_size,
    )

def chat_loop(
    generator: Llama,
    system_prompt: Optional[str] = None,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_gen_len: Optional[int] = None,
):
    """Run an interactive chat loop with the model."""
    # Keep system prompt separate from conversation history
    system_message = RawMessage(role="system", content=system_prompt) if system_prompt else None
    
    print("\nWelcome to Llama 3 Chat! Type 'quit' to exit, 'clear' to start a new conversation.")
    print("You can also type 'system: <prompt>' to set a new system prompt.\n")
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Handle special commands
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'clear':
            print("\nConversation cleared!")
            continue
        elif user_input.lower().startswith('system:'):
            system_prompt = user_input[7:].strip()
            system_message = RawMessage(role="system", content=system_prompt)
            print("\nSystem prompt updated!")
            continue
        
        # Create current conversation with just system prompt (if any) and current message
        current_messages = []
        if system_message:
            current_messages.append(system_message)
        current_messages.append(RawMessage(role="user", content=user_input))
        
        # Get model's response with just current message
        result = generator.chat_completion(
            current_messages,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        
        # Print assistant's response
        assistant_message = result.generation
        print(f"\nAssistant: {assistant_message.content}")

def main():
    # Get checkpoint directory from environment or use default
    ckpt_dir = os.getenv('CHECKPOINT_DIR', os.path.expanduser('~/.llama/checkpoints/Llama3.2-3B'))
    
    # Initialize the model
    print("Initializing Llama 3 model...")
    generator = setup_model(ckpt_dir)
    
    # Optional: Set a default system prompt
    default_system_prompt = "You are a helpful, respectful and honest assistant. Always provide accurate information and admit when you're not sure about something."
    
    # Start the chat loop
    try:
        chat_loop(generator, system_prompt=default_system_prompt)
    except KeyboardInterrupt:
        print("\nGoodbye!")

if __name__ == "__main__":
    main() 