from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class TranscriptReference(BaseModel):
    """A reference to a specific part of the interview transcript."""
    interview_question: str
    interview_response: str
    timestamp: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            'interview_question': self.interview_question,
            'interview_response': self.interview_response,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'TranscriptReference':
        ts = d.get('timestamp')
        return cls(
            interview_question=d['interview_question'],
            interview_response=d['interview_response'],
            timestamp=datetime.fromisoformat(ts) if ts else None
        )


class Memory(BaseModel):
    """Model for storing memories with their associated questions.

    Memories are aggregated: similar information from different parts of the
    transcript is combined into a single memory with multiple transcript_references.
    """
    id: str
    title: str
    text: str
    subtopic_links: List[Dict[str, Any]]
    metadata: dict
    timestamp: datetime
    transcript_references: List[TranscriptReference] = []
    question_ids: List[str] = []  # IDs of questions that generated this memory

    def to_dict(self) -> dict:
        """Convert Memory object to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'text': self.text,
            'subtopic_links': self.subtopic_links,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'transcript_references': [ref.to_dict() for ref in self.transcript_references],
            'question_ids': self.question_ids
        }

    def to_xml(self, include_source: bool = False, include_memory_info: bool = True) -> str:
        """Convert memory to XML format string.

        Args:
            include_source: Whether to include transcript references
        Returns:
            str: XML formatted string of the memory
        """
        lines = [
            '<memory>',
            f'<id>{self.id}</id>'
        ]

        if include_memory_info:
            lines.append(f'<title>{self.title}</title>')
            lines.append(f'<summary>{self.text}</summary>')
            lines.append(f'<subtopic_links>{self.subtopic_links}</subtopic_links>')

        if include_source and self.transcript_references:
            lines.append('<transcript_references>')
            for i, ref in enumerate(self.transcript_references):
                lines.append(f'<reference index="{i + 1}">')
                lines.append(
                    f'<interview_question>\n'
                    f'{ref.interview_question}\n'
                    f'</interview_question>'
                )
                lines.append(
                    f'<interview_response>\n'
                    f'{ref.interview_response}\n'
                    f'</interview_response>'
                )
                lines.append('</reference>')
            lines.append('</transcript_references>')

        lines.append('</memory>')
        return '\n'.join(lines)

    @classmethod
    def from_dict(cls, memory_dict: dict) -> 'Memory':
        """Create Memory object from dictionary. Backward-compatible with old format."""
        # Handle backward compatibility: old format had source_interview_question/response
        if 'transcript_references' in memory_dict:
            refs = [TranscriptReference.from_dict(r) for r in memory_dict['transcript_references']]
        elif 'source_interview_question' in memory_dict:
            refs = [TranscriptReference(
                interview_question=memory_dict['source_interview_question'],
                interview_response=memory_dict['source_interview_response'],
                timestamp=datetime.fromisoformat(memory_dict['timestamp'])
                    if 'timestamp' in memory_dict else None
            )]
        else:
            refs = []

        return cls(
            id=memory_dict['id'],
            title=memory_dict['title'],
            text=memory_dict['text'],
            subtopic_links=memory_dict['subtopic_links'],
            metadata=memory_dict['metadata'],
            timestamp=datetime.fromisoformat(memory_dict['timestamp']),
            transcript_references=refs,
            question_ids=memory_dict.get('question_ids', [])
        )

class MemorySearchResult(Memory):
    """Model for memory search results that includes similarity score."""
    similarity_score: float = Field(ge=0, le=1)  # Score between 0 and 1

    @classmethod
    def from_memory(cls, memory: Memory, similarity_score: float) -> 'MemorySearchResult':
        """Create a search result from a Memory object and similarity score."""
        return cls(
            id=memory.id,
            title=memory.title,
            text=memory.text,
            subtopic_links=memory.subtopic_links,
            metadata=memory.metadata,
            timestamp=memory.timestamp,
            transcript_references=memory.transcript_references,
            question_ids=memory.question_ids,
            similarity_score=similarity_score
        )
