"""
Interview Topic Manager

This module defines the InterviewTopicManager class, which tracks core topics with their completion status.
"""

import os
from typing import ClassVar, Dict, List, Optional, Tuple, Union, Any

import faiss
import numpy as np
from pydantic import BaseModel, Field
from src.content.embeddings.embedding_service import EmbeddingService
from src.content.question_bank.question import InterviewQuestion, Rubric
from src.content.session_agenda.core_topic import CoreTopic, SubTopic, EmergentInsight
from src.content.session_agenda.topic_evaluator import TopicEvaluator, MinimumThresholdSubtopicsEvaluator, get_registry

class InterviewTopicManager(BaseModel):
    """Tracks core topics with their completion status."""
    core_topic_dict: Dict[str, CoreTopic] = Field(default_factory=dict)
    interview_evaluator: TopicEvaluator = Field(default_factory=MinimumThresholdSubtopicsEvaluator)
    model_config = {
        "arbitrary_types_allowed": True
    }
    active_topic_id_list: List[str] = Field(default_factory=list)
    
    # Statistics only for recording
    coverage_stats: Dict[str, Any] = Field(default_factory=list)
    enable_emergent_subtopics: bool = False

    # Queue of task names waiting for the next deep dive batch
    pending_task_deep_dives: List[str] = Field(default_factory=list)

    # Embedding-based dedup for emergent subtopics
    embedding_service: Optional[Any] = Field(default=None, exclude=True)
    subtopic_embeddings: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    similarity_threshold: float = 0.7
    
    def __iter__(self):
        """Iterate over all core topics."""
        return iter(self.core_topic_dict.values())
    
    def __len__(self):
        """Return the number of core topics."""
        return len(self.core_topic_dict)
    
    def __contains__(self, topic_id: str) -> bool:
        """
        Implements the 'in' operator to check if a topic_id
        exists in the core_topic_dict.
        """
        return topic_id in self.core_topic_dict
    
    def __str__(self) -> str:
        """Returns a tree visualization of topics and questions.
        
        Example output:
        Topics
        ├── General
        │   └── How old are you?
        ├── Professional
        │   ├── How did you choose your career path?
        │   └── What specific rare plant species did you cultivate?
        │       └── Did you face any challenges?
        └── Personal
            └── Where did you grow up?
        """
        # TODO: This is unverified

        lines = ["Topics"]        
        def add_question_prefix(question: InterviewQuestion, 
                         prefix: str, is_last: bool) -> None:
            # Add the current question
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{question.question}")
            
            # Handle sub-questions
            if question.sub_questions:
                new_prefix = prefix + ("    " if is_last else "│   ")
                sub_questions = question.sub_questions
                for i, sub_q in enumerate(sub_questions):
                    add_question_prefix(sub_q, new_prefix, i == len(sub_questions) - 1)
        
        # Process each topic
        for topic_idx, core_topic in enumerate(self.core_topic_dict.values()):
            # Add topic
            topic_prefix = "└── " if topic_idx == len(self.core_topic_dict) - 1 else "├── "
            lines.append(f"{topic_prefix}{core_topic.description}")
            
            for subtopic_idx, subtopic in enumerate(core_topic):
                # Add topic
                subtopic_prefix = "    └── " if subtopic_idx == len(core_topic) - 1 else "    ├── "
                lines.append(f"{subtopic_prefix}{subtopic.description}")
            
                # Process questions under this topic
                question_prefix = "        " if subtopic_idx == len(subtopic) - 1 else "│       "
                for q_idx, question in enumerate(subtopic):
                    add_question_prefix(question, question_prefix, q_idx == len(subtopic) - 1)
        
        return "\n".join(lines)
    
    @classmethod
    def init_from_interview_plan(cls, interview_plan: List[Dict[str, Any]] = [],
                                 interview_evaluator: Optional[str] = None):
        # Initialize InterviewTopicManager using the interview plan provided
        manager = cls()
        for i, topic_dict in enumerate(interview_plan):
            topic_id = str(i + 1)
            
            curr_core_topic = CoreTopic(
                topic_id=topic_id,
                description=topic_dict['topic'],
                required_subtopics={},
                emergent_subtopics={},
                keywords=[],
                allow_emergent=topic_dict.get('allow_emergent', True),
                allow_strategic_planner=topic_dict.get('allow_strategic_planner', True),
                priority_weight=topic_dict.get('priority_weight', 1.0),
            )

            for j, subtopic in enumerate(topic_dict.get('subtopics', [])):
                subtopic_id = f"{topic_id}.{j + 1}"
                if isinstance(subtopic, dict):
                    subtopic_description = subtopic.get('description', subtopic.get('subtopic_description', ''))
                    subtopic_criteria = subtopic.get('coverage_criteria', [])
                else:
                    subtopic_description = subtopic
                    subtopic_criteria = []
                subtopic_priority = subtopic.get('priority_weight', 1.0) if isinstance(subtopic, dict) else 1.0
                subtopic_max_followups = subtopic.get('max_followups', None) if isinstance(subtopic, dict) else None
                curr_subtopic = SubTopic(
                    subtopic_id=subtopic_id,
                    core_topic_id=topic_id,
                    description=subtopic_description,
                    questions=[],
                    is_covered=False,
                    coverage_criteria=subtopic_criteria,
                    priority_weight=subtopic_priority,
                    max_followups=subtopic_max_followups,
                )
                curr_core_topic.add_required_subtopic(curr_subtopic)

            manager.add_core_topic(curr_core_topic)

        # Activate the 2 highest-priority topics initially
        sorted_topics = sorted(
            manager.core_topic_dict.values(),
            key=lambda t: t.priority_weight,
            reverse=True
        )
        for topic in sorted_topics[:2]:
            manager.add_topic_id_as_active_topic(topic.topic_id)
        
        # Register evaluator
        evaluator_registry = get_registry()
        if interview_evaluator:
            manager.interview_evaluator = evaluator_registry.create(interview_evaluator)
        else:
            manager.interview_evaluator = evaluator_registry.create("minimum_threshold")
            
        return manager
    
    def add_topic_id_as_active_topic(self, core_topic_id: str):
        if core_topic_id in self.core_topic_dict and core_topic_id not in self.active_topic_id_list:
            self.active_topic_id_list.append(core_topic_id)
    
    def add_core_topic(self, core_topic: CoreTopic):
        self.core_topic_dict[core_topic.topic_id] = core_topic

    def _is_task_deep_dive(self, topic_id: str) -> bool:
        """Return True if the topic is a Task Deep Dive topic."""
        topic = self.core_topic_dict.get(topic_id)
        return topic is not None and topic.description.startswith("Task Deep Dive:")

    MAX_CONCURRENT_DEEP_DIVES: ClassVar[int] = 3

    def _count_incomplete_deep_dives(self) -> int:
        """Return the number of incomplete Task Deep Dive topics."""
        return sum(
            1 for tid in self.core_topic_dict
            if self._is_task_deep_dive(tid) and not self.check_core_topic_completion(tid)
        )

    def add_task_deep_dive(self, task_name: str, subtopics: List[Dict[str, Any]]) -> str:
        """Add a Task Deep Dive topic for task_name, or queue it if the batch is full.

        Returns:
            "created:<topic_id>" if the topic was created immediately.
            "queued" if the max concurrent deep dives are already in progress.
            "exists" if a deep dive for this task already exists.
        """
        # Skip if already exists
        description = f"Task Deep Dive: {task_name}"
        if any(t.description == description for t in self.core_topic_dict.values()):
            return "exists"

        if self._count_incomplete_deep_dives() >= self.MAX_CONCURRENT_DEEP_DIVES:
            if task_name not in self.pending_task_deep_dives:
                self.pending_task_deep_dives.append(task_name)
            return "queued"

        topic_id = self.add_new_core_topic(description=description, subtopics=subtopics)
        return f"created:{topic_id}"

    def add_new_core_topic(self, description: str, subtopics: List[Dict[str, Any]]) -> str:
        """Dynamically add a new core topic with given subtopics. Returns the new topic_id."""
        topic_id = str(len(self.core_topic_dict) + 1)
        # Ensure uniqueness in case of collisions
        while topic_id in self.core_topic_dict:
            topic_id = str(int(topic_id) + 1)

        new_topic = CoreTopic(
            topic_id=topic_id,
            description=description,
            required_subtopics={},
            emergent_subtopics={},
            keywords=[],
        )

        for j, subtopic in enumerate(subtopics):
            subtopic_id = f"{topic_id}.{j + 1}"
            if isinstance(subtopic, dict):
                subtopic_description = subtopic.get('description', '')
                subtopic_criteria = subtopic.get('coverage_criteria', [])
            else:
                subtopic_description = str(subtopic)
                subtopic_criteria = []
            new_subtopic = SubTopic(
                subtopic_id=subtopic_id,
                core_topic_id=topic_id,
                description=subtopic_description,
                questions=[],
                is_covered=False,
                coverage_criteria=subtopic_criteria,
            )
            new_topic.add_required_subtopic(new_subtopic)

        self.add_core_topic(new_topic)
        self.add_topic_id_as_active_topic(topic_id)
        return topic_id
        
    def get_core_topic(self, core_topic_id: str) -> Optional[CoreTopic]:
        return self.core_topic_dict.get(core_topic_id, None)
    
    def _get_embedding_service(self) -> Optional[EmbeddingService]:
        """Get or lazily create the embedding service."""
        if self.embedding_service is None:
            self.embedding_service = EmbeddingService()
        return self.embedding_service

    def _ensure_subtopic_embeddings(self):
        """Compute and cache embeddings for any subtopics not yet in the cache."""
        service = self._get_embedding_service()
        if service is None or service.is_noop():
            return

        missing_ids = []
        missing_descriptions = []
        for core_topic in self.core_topic_dict.values():
            for subtopic in core_topic:
                if subtopic.subtopic_id not in self.subtopic_embeddings:
                    missing_ids.append(subtopic.subtopic_id)
                    missing_descriptions.append(subtopic.description)

        if missing_descriptions:
            embeddings = service.get_embeddings_batch(missing_descriptions)
            for sid, emb in zip(missing_ids, embeddings):
                self.subtopic_embeddings[sid] = emb

    def _check_duplicate_subtopic(self, description: str) -> Tuple[bool, float]:
        """Check if a subtopic description is too similar to any existing subtopic.

        Returns:
            Tuple of (is_duplicate, similarity_score).
        """
        service = self._get_embedding_service()
        if service is None or service.is_noop():
            return False, None, 0.0

        self._ensure_subtopic_embeddings()

        if not self.subtopic_embeddings:
            return False, None, 0.0

        # Build FAISS index from cached embeddings
        dim = service.get_embedding_dimension()
        index = faiss.IndexFlatL2(dim)

        id_list = list(self.subtopic_embeddings.keys())
        embedding_matrix = np.array(
            [self.subtopic_embeddings[sid] for sid in id_list]
        ).astype(np.float32)
        index.add(embedding_matrix)

        # Compute embedding for the new description
        new_embedding = service.get_embedding(description).reshape(1, -1).astype(np.float32)

        # Search for nearest neighbor
        distances, indices = index.search(new_embedding, 1)

        if indices[0][0] == -1:
            return False, None, 0.0

        # Convert L2 distance to similarity score (consistent with codebase pattern)
        similarity = float(1 / (1 + distances[0][0]))

        is_duplicate = similarity >= self.similarity_threshold
        return is_duplicate, similarity

    def add_emergent_subtopic(self, core_topic_id: str, new_subtopic_description: str) -> bool:
        core_topic = self.get_core_topic(core_topic_id)
        if core_topic is None:
            return False

        if not core_topic.allow_emergent:
            return False

        # Embedding-based deduplication check
        is_duplicate, similarity = self._check_duplicate_subtopic(new_subtopic_description)
        if is_duplicate:
            return False

        subtopic_list_length = len(core_topic)
        subtopic_id = f"{core_topic_id}.{subtopic_list_length + 1}"
        new_subtopic = SubTopic(
            subtopic_id=subtopic_id,
            core_topic_id=core_topic_id,
            description=new_subtopic_description,
            questions=[],
            is_covered=False
        )
        added = core_topic.add_emergent_subtopic(new_subtopic)

        # Cache embedding for the newly added subtopic
        if added:
            service = self._get_embedding_service()
            if service is not None and not service.is_noop():
                self.subtopic_embeddings[subtopic_id] = service.get_embedding(new_subtopic_description)

        return added

    def add_question(self, subtopic_id: str, new_question: InterviewQuestion) -> bool:
        """Add a question to a core topic then to a subtopic. Returns True is adding succeeded."""
        topic_id = subtopic_id.split(".")[0]
        core_topic = self.get_core_topic(topic_id)
        if core_topic is None:
            return False
        
        subtopic = core_topic.get_subtopic(subtopic_id)
        if subtopic is None:
            return False
        
        return subtopic.add_question(new_question)
    
    def add_note_to_subtopic(self, subtopic_id: str, note: str) -> bool:
        topic_id = subtopic_id.split(".")[0]
        core_topic = self.get_core_topic(topic_id)
        if core_topic is None:
            return False
        
        subtopic = core_topic.get_subtopic(subtopic_id)
        if subtopic is None:
            return False
        
        return subtopic.add_note(note)
    
    def get_question(self, topic_id: str, subtopic_id: str, question_id: str) -> Optional[InterviewQuestion]:
        core_topic = self.get_core_topic(topic_id)
        if core_topic is None:
            return None
        
        subtopic = core_topic.get_subtopic(subtopic_id)
        if subtopic is None:
            return None
        
        return subtopic.get_question(question_id)
    
    def reset(self):
        """Clear all non-default elements inside CoreTopic and SubTopic."""
        for core_topic in self.core_topic_dict.values():
            core_topic.reset()
            
    def check_core_topic_completion(self, core_topic_id: str) -> bool:
        """Check the completion of a core topic."""
        core_topic = self.get_core_topic(core_topic_id)
        if core_topic is None:
            return False
        
        return self.interview_evaluator.is_complete(core_topic)
    
    def check_core_topic_score(self, core_topic_id: str) -> float:
        """Check the score of a core topic."""
        core_topic = self.get_core_topic(core_topic_id)
        if core_topic is None:
            return False
        
        return self.interview_evaluator.get_coverage_score(core_topic)
    
    def check_all_core_topic_completion(self) -> bool:
        """Check the completion of all core topics."""
        return all(self.check_core_topic_completion(topic_id) for topic_id in self.core_topic_dict.keys())
    
    def get_all_incomplete_core_topic(self) -> List[CoreTopic]:
        """Get all incomplete core topics."""
        core_topic_list = []
        for core_topic in self.core_topic_dict.values():
            if not self.check_core_topic_completion(core_topic.topic_id):
                core_topic_list.append(core_topic)
            
        return core_topic_list
    
    def any_active_topic_allows_strategic_planner(self) -> bool:
        """Return True if at least one active topic has allow_strategic_planner=True."""
        for topic_id in self.active_topic_id_list:
            topic = self.get_core_topic(topic_id)
            if topic is not None and topic.allow_strategic_planner:
                return True
        return False

    def any_active_topic_allows_emergent(self) -> bool:
        """Return True if at least one active topic has allow_emergent=True.

        Used to gate prompt instructions that encourage the interviewer to
        free-associate beyond the configured subtopics (emergent-insight
        probing). When all active topics opt out, the interviewer should
        stick strictly to the configured agenda.
        """
        for topic_id in self.active_topic_id_list:
            topic = self.get_core_topic(topic_id)
            if topic is not None and topic.allow_emergent:
                return True
        return False

    def use_emergent_subtopics(self):
        self.enable_emergent_subtopics = True
    
    def get_active_topics(self) -> List[CoreTopic]:
        """Get all current active topics."""
        if self.enable_emergent_subtopics:
            required_only = False
        else:
            required_only = True
        
        list_active_topics = []
        for topic_id in self.active_topic_id_list:
            copy_topic = CoreTopic.get_topic_with_active_subtopics(self.get_core_topic(topic_id), required_only=required_only)
            list_active_topics.append(copy_topic)
        return list_active_topics
    
    def get_all_topics(self) -> List[CoreTopic]:
        """Get all topics."""
        if self.enable_emergent_subtopics:
            required_only = False
        else:
            required_only = True
            
        list_core_topics = []
        for core_topic in self.core_topic_dict.values():
            copy_topic = CoreTopic.get_copy_of_core_topic(core_topic, required_only=required_only)
            list_core_topics.append(copy_topic)
            
        return list_core_topics
    
    def update_subtopic_coverage(self, subtopic_id: str, aggregated_notes: str) -> bool:
        topic_id = subtopic_id.split(".")[0]
        core_topic = self.get_core_topic(topic_id)
        if core_topic is None:
            return False
        
        subtopic = core_topic.get_subtopic(subtopic_id)
        if subtopic is None:
            return False
        
        return subtopic.update_coverage_with_summary(aggregated_notes=aggregated_notes)
    
    def add_emergent_insight_subtopic(self, subtopic_id: str, insight_data: Dict[str, Any]) -> bool:
        topic_id = subtopic_id.split(".")[0]
        core_topic = self.get_core_topic(topic_id)
        if core_topic is None:
            return False
        
        subtopic = core_topic.get_subtopic(subtopic_id)
        if subtopic is None:
            return False
        
        insight = EmergentInsight.from_dict(insight_data)
        return subtopic.add_emergent_insight(insight=insight)
    
    def update_subtopic_criteria_coverage(self, subtopic_id: str, statuses: list) -> bool:
        topic_id = subtopic_id.split(".")[0]
        core_topic = self.get_core_topic(topic_id)
        if core_topic is None:
            return False
        subtopic = core_topic.get_subtopic(subtopic_id)
        if subtopic is None:
            return False
        subtopic.update_criteria_coverage(statuses)
        return True

    def give_feedback_subtopic_coverage(self, subtopic_id: str, feedback: str) -> bool:
        topic_id = subtopic_id.split(".")[0]
        core_topic = self.get_core_topic(topic_id)
        if core_topic is None:
            return False
        
        subtopic = core_topic.get_subtopic(subtopic_id)
        if subtopic is None:
            return False
        
        return subtopic.update_coverage_feedback_gap(feedback=feedback)
    
    def revise_agenda_after_update(self):
        deep_dive_enabled = os.getenv("ENABLE_TASK_DEEP_DIVE", "false").lower() == "true"
        if not deep_dive_enabled:
            # Strip persisted deep dive topics when feature is off
            self.pending_task_deep_dives = []
            dd_ids = [tid for tid in self.core_topic_dict if self._is_task_deep_dive(tid)]
            for tid in dd_ids:
                del self.core_topic_dict[tid]
        elif self.pending_task_deep_dives:
            # Fill up to MAX_CONCURRENT_DEEP_DIVES from the queue
            from src.agents.session_scribe.tools import TASK_DEEP_DIVE_SUBTOPICS
            while (self.pending_task_deep_dives
                   and self._count_incomplete_deep_dives() < self.MAX_CONCURRENT_DEEP_DIVES):
                next_task = self.pending_task_deep_dives.pop(0)
                self.add_task_deep_dive(task_name=next_task, subtopics=TASK_DEEP_DIVE_SUBTOPICS)

        # Build active topic list — deep dive topics take priority over regular topics.
        incomplete_deep_dives = [
            tid for tid in self.core_topic_dict
            if self._is_task_deep_dive(tid) and not self.check_core_topic_completion(tid)
        ]

        self.active_topic_id_list = []
        if incomplete_deep_dives:
            # All incomplete deep dives in the current batch are active
            self.active_topic_id_list = list(incomplete_deep_dives)
        else:
            # No deep dive in progress — activate up to 3 highest-priority incomplete topics
            sorted_topics = sorted(
                self.core_topic_dict.values(),
                key=lambda t: t.priority_weight,
                reverse=True
            )
            for topic in sorted_topics:
                if not self.check_core_topic_completion(topic.topic_id):
                    self.active_topic_id_list.append(topic.topic_id)
                if len(self.active_topic_id_list) == 3:
                    break

        # Stats gathering only
        curr_coverage_stats = {}
        for topic_id, core_topic in self.core_topic_dict.items():
            curr_coverage_stats[topic_id] = self.interview_evaluator.get_all_statistics(core_topic)
        self.coverage_stats = curr_coverage_stats

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'core_topic_dict': {k: v.to_dict() for k, v in self.core_topic_dict.items()},
            'active_topic_id_list': self.active_topic_id_list,
            'coverage_stats': self.coverage_stats, # Stats only
            'pending_task_deep_dives': self.pending_task_deep_dives,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'InterviewTopicManager':
        """Create from dictionary."""
        # Assuming CoreTopic has a from_dict method, adjust as needed
        manager = cls()

        core_topic_dict_data = data.get("core_topic_dict", {})
        active_topic_id_list_data = data.get("active_topic_id_list", [])
        for topic_id, core_topic_data in core_topic_dict_data.items():
            # CoreTopic.from_dict should reconstruct its SubTopics and Questions
            core_topic = CoreTopic.from_dict(core_topic_data)
            manager.add_core_topic(core_topic)
            
        for topic_id in active_topic_id_list_data:
            manager.add_topic_id_as_active_topic(topic_id)

        manager.coverage_stats = data.get("coverage_stats", {})
        manager.pending_task_deep_dives = data.get("pending_task_deep_dives", [])
        return manager
