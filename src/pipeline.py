import json

from question_classifier import classify_question
from retriever import AudioContextRetriever
from answer_generator import AudioAnswerGenerator


class AudioContextPipeline:
    """
    End-to-end Audio Context Layer.

    Question
        ↓
    Question Classifier
        ↓
    Context Retriever
        ↓
    Answer Generator
        ↓
    Final Answer
    """

    def __init__(
        self,
        contexts_dir="data/processed/contexts"
    ):

        self.retriever = AudioContextRetriever(
            contexts_dir
        )

        self.answer_generator = AudioAnswerGenerator()

    def answer(self, question, context_id):

        # ==============================================
        # 1. QUESTION CLASSIFICATION
        # ==============================================

        question_type = classify_question(
            question
        )

        # ==============================================
        # 2. LOAD AUDIO CONTEXT
        # ==============================================

        context = self.retriever.load_context(
            context_id
        )

        # ==============================================
        # 3. RETRIEVE RELEVANT INFORMATION
        # ==============================================

        retrieval_result = self.retriever.retrieve(
            question,
            context
        )

        relevant_context = retrieval_result[
            "relevant_context"
        ]

        # ==============================================
        # 4. GENERATE ANSWER
        # ==============================================

        answer = self.answer_generator.generate(
            question,
            relevant_context
        )

        # ==============================================
        # RETURN COMPLETE RESULT
        # ==============================================

        return {
            "context_id": context_id,
            "question": question,
            "question_type": question_type,
            "retrieved_context": relevant_context,
            "answer": answer
        }


# ======================================================
# INTERACTIVE TEST
# ======================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AUDIO CONTEXT LAYER - END-TO-END PIPELINE")
    print("=" * 70)

    pipeline = AudioContextPipeline()

    context_id = "context_0001"

    test_questions = [

        "What sounds are present in the audio?",

        "How many bird sounds are there?",

        "How many dog sounds are there?",

        "How long did the bird sound last?",

        "What happened before the dog sound?",

        "What happened after the bird sound?",

        "What is the order of the sounds?",

        "What happened first?",

        "What happened last?",

        "Why did the dog sound occur?"
    ]

    for question in test_questions:

        print()
        print("=" * 70)
        print(f"QUESTION: {question}")
        print("=" * 70)

        try:

            result = pipeline.answer(
                question,
                context_id
            )

            print(
                f"Question Type: "
                f"{result['question_type']}"
            )

            print()

            print(
                "Retrieved Context:"
            )

            print(
                json.dumps(
                    result["retrieved_context"],
                    indent=2
                )
            )

            print()

            print(
                f"FINAL ANSWER:\n"
                f"{result['answer']}"
            )

        except Exception as error:

            print(
                f"ERROR: {error}"
            )

    print()
    print("=" * 70)
    print("END-TO-END PIPELINE TEST COMPLETED")
    print("=" * 70)