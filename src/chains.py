from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import USE_OFFLINE_MODEL, OLLAMA_MODEL, FINAL_TOP_N

class ClassificationResult(BaseModel):
    is_on_topic: bool = Field(description="True if query is a movie recommendation request, False otherwise")
    query_case: Optional[int] = Field(description="Integer 1 through 6 representing the query case. None if on-topic but doesn't fit cases.")
    cleaned_text: Optional[str] = Field(description="Cleaned text for vague queries or noisy summaries")
    referenced_titles: List[str] = Field(description="Exact titles mentioned. For multiple titles, select the 2 most relevant.")
    constraint_text: Optional[str] = Field(description="Modifiers or constraints like 'but more emotional'")

class ReRankResult(BaseModel):
    ranked_titles: List[str] = Field(description="List of candidate movie titles sorted by relevance to the query")

def get_llm():
    if USE_OFFLINE_MODEL:
        return ChatOllama(model=OLLAMA_MODEL, temperature=0.0)
    else:
        return ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.0)

def build_classifier_chain():
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Classify movie recommendation queries into one of 6 cases.
1: Noisy / typo summary. (Set cleaned_text)
2: Free-text description. (Set cleaned_text)
3: Direct title match. (Set referenced_titles)
4: Title + modifier. (Set referenced_titles and constraint_text)
5: Multiple titles. (Set referenced_titles to max 2 most relevant)
6: Multiple titles + modifier. (Set referenced_titles to max 2 most relevant, and constraint_text)

Available exact titles:
{titles}

CRITICAL: You must aggressively fix typos and fuzzy match user input to the EXACT titles from the list (e.g. 'incption' -> 'Inception', 'th matrxx' -> 'The Matrix'). If a user mentions a movie, ALWAYS populate referenced_titles with the perfectly spelled exact match from the list above. Do NOT include movies in the recommendation response if they are in referenced_titles.
"""),
        ("human", "{query}")
    ])
    return prompt | llm.with_structured_output(ClassificationResult)

def build_hyde_chain():
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate a detailed hypothetical movie summary that perfectly matches the user's intent. Do not mention that this is a hypothetical summary."),
        ("human", "Cleaned Text: {cleaned_text}\nReferenced Movies:\n{movie_summaries}\nConstraints: {constraint_text}")
    ])
    return prompt | llm | StrOutputParser()

def build_generation_chain():
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a helpful movie recommendation assistant. You will be provided with up to 10 candidate movies and their short summaries. Select a minimum of 3 and a maximum of {FINAL_TOP_N} most relevant movies from these candidates. Write a conversational recommendation response for the user explaining why these specific movies fit their request. NEVER invent or recommend a title not explicitly listed in the candidates. Constraints must be respected."),
        ("human", "Original Query: {query}\nConstraints: {constraint_text}\nCandidates:\n{candidates}")
    ])
    return prompt | llm | StrOutputParser()
