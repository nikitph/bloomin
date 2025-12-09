"""
Section Extractor

Extracts structured LocalSections from raw text by identifying:
1. Noun phrases (objects/entities)
2. Their associated modifiers (adjectives, attributes)
3. Binding them into coherent sections

This bridges the gap between raw text and the topos machinery.
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from .logic import LocalSection, Proposition


@dataclass
class ExtractedEntity:
    """An entity extracted from text with its properties."""
    entity_type: str  # noun (car, bike, gun, etc.)
    modifiers: List[str]  # adjectives/attributes (red, fast, fake, etc.)
    span: Tuple[int, int]  # character positions in original text
    raw_text: str  # the original phrase


class SectionExtractor:
    """
    Extracts structured sections from natural language text.

    Uses pattern matching and simple NLP to identify:
    - Noun phrases with their modifiers
    - Property bindings (which adjectives go with which nouns)
    """

    # Common nouns we care about (expandable)
    NOUNS = {
        'car', 'cars', 'bike', 'bikes', 'bicycle', 'bicycles',
        'gun', 'guns', 'weapon', 'weapons', 'firearm', 'firearms',
        'ring', 'rings', 'diamond', 'diamonds', 'jewelry',
        'phone', 'phones', 'watch', 'watches',
        'house', 'houses', 'building', 'buildings',
        'dog', 'dogs', 'cat', 'cats', 'bird', 'birds', 'lion', 'lions',
        'person', 'people', 'man', 'men', 'woman', 'women', 'child', 'children',
        'food', 'fruit', 'vegetable', 'meal',
        'vehicle', 'vehicles', 'truck', 'trucks', 'motorcycle', 'motorcycles',
        'computer', 'computers', 'laptop', 'laptops',
        'book', 'books', 'document', 'documents',
        'weather', 'rain', 'sun', 'snow', 'storm',
        'object', 'objects', 'item', 'items', 'thing', 'things',
    }

    # Common adjectives/modifiers
    MODIFIERS = {
        # Colors
        'red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'purple',
        'pink', 'brown', 'gray', 'grey', 'crimson', 'golden', 'silver',
        # Size
        'big', 'small', 'large', 'tiny', 'huge', 'little', 'tall', 'short',
        # Quality
        'good', 'bad', 'great', 'terrible', 'excellent', 'poor',
        'new', 'old', 'ancient', 'modern', 'young',
        'fast', 'slow', 'quick', 'speedy',
        'hot', 'cold', 'warm', 'cool',
        'heavy', 'light', 'lightweight',
        # Cost/Value
        'cheap', 'expensive', 'costly', 'affordable', 'pricey',
        'free', 'valuable', 'worthless',
        # Authenticity (important for modifier detection)
        'real', 'fake', 'authentic', 'genuine', 'artificial', 'synthetic',
        'true', 'false', 'imitation', 'replica', 'counterfeit',
        'original', 'copy', 'duplicate',
        # Material
        'plastic', 'metal', 'wooden', 'glass', 'stone', 'steel', 'iron',
        'leather', 'rubber', 'cotton', 'silk', 'gold', 'platinum',
        # State
        'broken', 'working', 'functional', 'damaged', 'intact',
        'clean', 'dirty', 'wet', 'dry',
        'safe', 'dangerous', 'lethal', 'harmless',
        'alive', 'dead',
        # Weather
        'sunny', 'rainy', 'cloudy', 'stormy', 'clear', 'foggy',
        # Other
        'beautiful', 'ugly', 'pretty', 'handsome',
        'electric', 'manual', 'automatic',
        'healthy', 'unhealthy', 'fresh', 'rotten',
    }

    # Patterns for noun phrases: (modifier* noun)
    # We'll use a simple approach: find nouns and look backwards for modifiers

    def __init__(self, custom_nouns: Optional[Set[str]] = None,
                 custom_modifiers: Optional[Set[str]] = None):
        self.nouns = self.NOUNS.copy()
        self.modifiers = self.MODIFIERS.copy()

        if custom_nouns:
            self.nouns.update(custom_nouns)
        if custom_modifiers:
            self.modifiers.update(custom_modifiers)

    def _tokenize(self, text: str) -> List[Tuple[str, int, int]]:
        """Tokenize text, returning (token, start, end) tuples."""
        tokens = []
        for match in re.finditer(r'\b\w+\b', text.lower()):
            tokens.append((match.group(), match.start(), match.end()))
        return tokens

    def _extract_noun_phrases(self, text: str) -> List[ExtractedEntity]:
        """Extract noun phrases with their modifiers."""
        tokens = self._tokenize(text)
        entities = []

        # Skip words (articles, conjunctions, etc.)
        skip_words = {'the', 'a', 'an', 'and', 'or', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}

        i = 0
        while i < len(tokens):
            token, start, end = tokens[i]

            # Check if this is a noun
            if token in self.nouns:
                # Look backwards for modifiers, skipping articles/conjunctions
                modifiers = []
                phrase_start = start

                j = i - 1
                while j >= 0:
                    prev_token, prev_start, prev_end = tokens[j]

                    # Skip articles and conjunctions
                    if prev_token in skip_words:
                        j -= 1
                        continue

                    # Check if it's a modifier
                    if prev_token in self.modifiers:
                        modifiers.insert(0, prev_token)
                        phrase_start = prev_start
                        j -= 1
                    else:
                        # Stop if we hit a non-modifier, non-skip word
                        break

                # Also look forward for post-modifiers (less common in English)
                phrase_end = end

                raw = text[phrase_start:phrase_end]

                entities.append(ExtractedEntity(
                    entity_type=token,
                    modifiers=modifiers,
                    span=(phrase_start, phrase_end),
                    raw_text=raw
                ))

            i += 1

        return entities

    def _split_sentences(self, text: str) -> List[Tuple[str, int]]:
        """Split text into sentences with their starting positions."""
        sentences = []
        # Simple sentence splitting on . ! ?
        pattern = r'[^.!?]*[.!?]+'

        for match in re.finditer(pattern, text):
            sentences.append((match.group().strip(), match.start()))

        # Handle text without sentence-ending punctuation
        if not sentences:
            sentences.append((text.strip(), 0))

        return sentences

    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract all entities with their bound properties from text."""
        return self._extract_noun_phrases(text)

    def extract_sections(self, text: str, doc_id: str = "doc") -> List[LocalSection]:
        """
        Extract structured LocalSections from text.

        Each noun phrase becomes a section with propositions for:
        - The entity type (is_car, is_bike, etc.)
        - Each modifier (is_red, is_fast, etc.)

        Args:
            text: Raw text to parse
            doc_id: Document identifier for section naming

        Returns:
            List of LocalSections with bound propositions
        """
        entities = self.extract_entities(text)
        sections = []

        for i, entity in enumerate(entities):
            section_id = f"{doc_id}_entity_{i}"

            # Create propositions
            propositions = []

            # Entity type proposition
            # Normalize: bikes -> bike, cars -> car
            entity_norm = entity.entity_type.rstrip('s') if entity.entity_type.endswith('s') and len(entity.entity_type) > 2 else entity.entity_type

            propositions.append(Proposition(
                predicate=f"is_{entity_norm}",
                confidence=1.0,
                support={section_id}
            ))

            # Modifier propositions
            for mod in entity.modifiers:
                propositions.append(Proposition(
                    predicate=f"is_{mod}",
                    confidence=1.0,
                    support={section_id}
                ))

            section = LocalSection(
                region_id=section_id,
                witness_ids={section_id},
                propositions=propositions
            )
            sections.append(section)

        return sections

    def extract_sections_by_sentence(self, text: str, doc_id: str = "doc") -> List[LocalSection]:
        """
        Extract sections grouped by sentence.

        Entities in the same sentence share a section, capturing
        co-occurrence within sentence boundaries.
        """
        sentences = self._split_sentences(text)
        all_sections = []

        for sent_idx, (sentence, sent_start) in enumerate(sentences):
            entities = self.extract_entities(sentence)

            if not entities:
                continue

            section_id = f"{doc_id}_sent_{sent_idx}"
            propositions = []

            for entity in entities:
                # Entity type
                entity_norm = entity.entity_type.rstrip('s') if entity.entity_type.endswith('s') and len(entity.entity_type) > 2 else entity.entity_type

                propositions.append(Proposition(
                    predicate=f"is_{entity_norm}",
                    confidence=1.0,
                    support={section_id}
                ))

                # Modifiers
                for mod in entity.modifiers:
                    propositions.append(Proposition(
                        predicate=f"is_{mod}",
                        confidence=1.0,
                        support={section_id}
                    ))

            if propositions:
                section = LocalSection(
                    region_id=section_id,
                    witness_ids={section_id},
                    propositions=propositions
                )
                all_sections.append(section)

        return all_sections

    def query_to_predicates(self, query: str) -> Set[str]:
        """
        Convert a query string to a set of predicates to search for.

        "red bike" -> {"is_red", "is_bike"}
        """
        predicates = set()
        tokens = self._tokenize(query)

        for token, _, _ in tokens:
            if token in self.nouns:
                # Normalize plurals
                token_norm = token.rstrip('s') if token.endswith('s') and len(token) > 2 else token
                predicates.add(f"is_{token_norm}")
            elif token in self.modifiers:
                predicates.add(f"is_{token}")

        return predicates

    def find_matching_sections(
        self,
        sections: List[LocalSection],
        required_predicates: Set[str]
    ) -> List[LocalSection]:
        """
        Find sections that contain ALL required predicates.

        This is the compositional query - predicates must be
        bound together in the SAME section.
        """
        matches = []

        for section in sections:
            section_preds = {p.predicate for p in section.propositions}

            if required_predicates.issubset(section_preds):
                matches.append(section)

        return matches


def extract_and_query(text: str, query: str, doc_id: str = "doc") -> Dict:
    """
    Convenience function: extract sections from text and run a compositional query.

    Returns dict with sections, query predicates, and matches.
    """
    extractor = SectionExtractor()

    # Extract sections (one per noun phrase for precise binding)
    sections = extractor.extract_sections(text, doc_id)

    # Convert query to predicates
    query_preds = extractor.query_to_predicates(query)

    # Find matching sections
    matches = extractor.find_matching_sections(sections, query_preds)

    return {
        "text": text,
        "query": query,
        "sections": sections,
        "query_predicates": query_preds,
        "matches": matches,
        "has_match": len(matches) > 0
    }
