from langchain_huggingface import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import pipeline
from collections import Counter
import torch

def create_sentiment_chain():
    # Create the sentiment pipeline
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        device=0 if torch.cuda.is_available() else -1  # Use GPU if available
    )
    
    # Create HuggingFacePipeline LLM
    llm = HuggingFacePipeline(pipeline=sentiment_pipe)
    
    # Create prompt template
    prompt = PromptTemplate(
        input_variables=["text"],
        template="{text}"
    )
    
    # Create the chain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    return chain

def batch_analyze_sentiments(texts, batch_size=32):
    # Initialize the chain
    chain = create_sentiment_chain()
    
    results = []
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Process batch concurrently
        batch_results = chain.apply(
            [{"text": text} for text in batch]
        )
        results.extend([result['text'] for result in batch_results])
    
    # Count sentiments
    sentiment_counts = Counter(results)
    
    # Create summary
    summary = {
        'positive': sentiment_counts.get('positive', 0),
        'negative': sentiment_counts.get('negative', 0),
        'neutral': sentiment_counts.get('neutral', 0),
        'total_texts': len(texts)
    }
    
    return summary

# Example usage
if __name__ == "__main__":
    sample_texts = [
        "I love this product! It's amazing!",
        "This is terrible, don't buy it.",
        "It's okay, nothing special.",
        "The quality is excellent!",
        "I'm really disappointed with the service."
    ] * 20  # Multiple texts to demonstrate batch processing
    
    results = batch_analyze_sentiments(sample_texts)
    print("\nSentiment Analysis Summary:")
    print(f"Positive texts: {results['positive']}")
    print(f"Negative texts: {results['negative']}")
    print(f"Neutral texts: {results['neutral']}")
    print(f"Total texts analyzed: {results['total_texts']}")