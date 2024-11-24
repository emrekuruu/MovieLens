from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from typing import List

class FeatureBasedFeedback(BaseModel):
    """
    Structured output for movie feedback.
    """
    binary_rating: bool  # True for positive alignment, False for negative
    positive_genres: List[str]  # List of genres that positively align with preferences
    negative_genres: List[str]  # List of genres that negatively align with preferences
    feedback: str  # Simple feedback explanation

def get_feedback_prompt(enjoyed_movies, disliked_movies, movie_genres, query_movie):
    # Format the input movie data
    enjoyed_movies_formatted = "\n".join(
        [f"- {movie} - Genres: [ {', '.join(movie_genres[movie])} ]" for movie in enjoyed_movies]
    )
    disliked_movies_formatted = "\n".join(
        [f"- {movie} - Genres: [ {', '.join(movie_genres[movie])} ]" for movie in disliked_movies]
    )

    # Get genres for the query movie from the dictionary
    query_movie_genres = movie_genres.get(query_movie, [])

    # Define the prompt template
    feature_template = """
    
    Below is a list of movies that you have enjoyed and disliked. Each movie is listed along with its genres.

    Enjoyed Movies:

{enjoyed_movies}



    Disliked Movies:

{disliked_movies}

    Based on these preferences, analyze the recommended movie below and provide structured feedback in the following format:

    - Binary Rating: True or False (whether the movie aligns with your preferences).

    - Positive Genres: List of genres from the recommended movie that you like.

    - Negative Genres: List of genres from the recommended movie that you dislike.

    - Feedback: A simple explanation in the format:
        "[Movie Name] does/does not align with my preferences because I like/dislike [genres]."

    Recommended Movie:
    {query_movie}

    Examples:

    Recommended Movie: Interstellar (Sci-Fi, Drama)
    Feedback:
    - Binary Rating: True
    - Positive Genres: ["Sci-Fi"]
    - Negative Genres: ["Drama"]
    - Feedback: "Interstellar aligns with my preferences because I like Sci-Fi very much, even though i dont enjoy Drama too much."

    Recommended Movie: Twilight (Romance, Fantasy)
    Feedback:
    - Binary Rating: False
    - Positive Genres: []
    - Negative Genres: ["Romance", "Fantasy"]
    - Feedback: "Twilight does not align with my preferences because I dislike Romance and Fantasy."

    Now, provide feedback for the recommended movie based on the preferences provided.
    """

    # Create the LangChain PromptTemplate
    prompt_template = PromptTemplate(
        input_variables=["enjoyed_movies", "disliked_movies", "query_movie"],
        template=feature_template,
    )

    # Fill the template with input data
    filled_prompt = prompt_template.format(
        enjoyed_movies=enjoyed_movies_formatted,
        disliked_movies=disliked_movies_formatted,
        query_movie=f"{query_movie} ({', '.join(query_movie_genres)})"
    )

    return filled_prompt
