from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
import logging

class CountryLegalAgent:
    def __init__(self, country: str, vector_store, llm):
        self.country = country
        self.vector_store = vector_store
        self.llm = llm
        
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=3,
            return_messages=True
        )
        
        # Define the retriever
        self.retriever = self.vector_store.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Country-specific system prompt
        self.system_prompt = self._get_country_prompt()

        # Create the full retrieval and generation chain using LCEL
        self.retrieval_chain = self._create_retrieval_chain()
    
    def _create_retrieval_chain(self):
        """Create a retrieval chain using LangChain Expression Language (LCEL)."""
        
        # 1. Define the prompt template
        template = f"""
{self.system_prompt}

LEGAL QUESTION: {{question}}

RELEVANT LEGAL INFORMATION:
{{context}}

Please provide a comprehensive answer that:
1. Directly addresses the legal question.
2. References relevant laws, acts, or regulations from {self.country.title()}.
3. Explains the legal principles involved.
4. Provides practical guidance where appropriate.
5. Emphasizes that this is general legal information, not legal advice.

IMPORTANT: Always recommend consulting with a qualified {self.country.title()} attorney for specific legal matters.

Response:"""
        
        prompt = ChatPromptTemplate.from_template(template)

        # 2. Create the LCEL chain
        # This chain retrieves context, passes it along with the question,
        # formats them into a prompt, sends it to the LLM, and parses the output.
        chain = (
            RunnableParallel(
                context=(lambda x: self.format_docs(self.retriever.invoke(x["question"]))),
                question=RunnablePassthrough()
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain

    @staticmethod
    def format_docs(docs):
        """Helper function to format retrieved documents into a single string."""
        if not docs:
            return "No specific legal documents found."
        return "\n\n".join(f"Document {i+1}: {doc.page_content[:500]}..." for i, doc in enumerate(docs))

    def _get_country_prompt(self):
        """Get the country-specific part of the system prompt."""
        prompts = {
            "india": "You are a knowledgeable legal assistant specializing in Indian law. You have access to Indian legal documents, acts, and regulations.",
            "usa": "You are a knowledgeable legal assistant specializing in United States law. You have access to U.S. legal documents, federal and state laws, and regulations.",
            "germany": "You are a knowledgeable legal assistant specializing in German law. You have access to German legal documents, codes, and EU regulations, including the BGB, StGB, and EU directives."
        }
        return prompts.get(self.country, "You are a legal assistant.")
    
    def get_response(self, query: str):
        """Get response from the legal agent using the LCEL chain."""
        try:
            # The input to the chain is a dictionary with the 'question' key
            response = self.retrieval_chain.invoke({"question": query})
            
        
            response = self._format_response(response)
            
            return response
            
        except Exception as e:
            logging.error(f"Error in get_response: {e}")
            return f"I apologize, but I encountered an error while processing your legal question: {str(e)}. Please try rephrasing your question or contact support if the issue persists."
    
    def _format_response(self, response: str) -> str:
        """Format and clean the final response."""
        response = response.strip()
        disclaimer = f"\n\n**Legal Disclaimer**: This information is for general guidance only and does not constitute legal advice. Please consult with a qualified {self.country.title()} attorney for specific legal matters."
        
        if "disclaimer" not in response.lower() and "legal advice" not in response.lower():
            response += disclaimer
        
        return response
    
