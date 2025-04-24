import re
from typing import Tuple, Dict

class PIIMasker:
    def __init__(self):
        self.patterns = [
            {
                'entity': 'credit_debit_no',
                'pattern': re.compile(r'\b(?:\d{4}[ -]?){3}(?:\d{4})\b'),
                'priority': 1
            },
            {
                'entity': 'aadhar_num',
                'pattern': re.compile(r'\b\d{4}[ -]?\d{4}[ -]?\d{4}\b'),
                'priority': 2
            },
            {
                'entity': 'email',
                'pattern': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', re.IGNORECASE),
                'priority': 3
            },
            {
                'entity': 'phone_number',
                'pattern': re.compile(r'\b(?:\+?91[ -]?)?[6-9]\d{9}\b'),
                'priority': 4
            },
            {
                'entity': 'cvv_no',
                'pattern': re.compile(r'\b\d{3,4}\b'),
                'priority': 5
            },
            {
                'entity': 'expiry_no',
                'pattern': re.compile(r'\b(0[1-9]|1[0-2])(?:/?-?)(\d{2,4})\b'),
                'priority': 6
            },
            {
                'entity': 'dob',
                'pattern': re.compile(r'\b(0[1-9]|1[0-9]|2[0-9]|3[0-1])[-/](0[1-9]|1[0-2])[-/](\d{4})\b'),
                'priority': 7
            },
            {
                'entity': 'full_name',
                'pattern': re.compile(r'\b([A-Z][a-z]+)(\s+[A-Z][a-z]+)+\b'),
                'priority': 8
            },
        ]
        self.patterns.sort(key=lambda x: x['priority'])

    def mask(self, text: str) -> Tuple[str, Dict[str, str]]:
        matches = []
        
        # Find all matches with priority handling
        for entity_def in self.patterns:
            pattern = entity_def['pattern']
            entity = entity_def['entity']
            for match in pattern.finditer(text):
                start, end = match.span()
                overlap = False
                # Check for overlaps with higher priority matches
                for existing in matches:
                    if not (end <= existing['start'] or start >= existing['end']):
                        overlap = True
                        break
                if not overlap:
                    matches.append({
                        'start': start,
                        'end': end,
                        'text': match.group(),
                        'entity': entity
                    })

        # Sort matches by occurrence and assign tokens
        matches.sort(key=lambda x: x['start'])
        counters = {entity['entity']: 0 for entity in self.patterns}
        for match in matches:
            entity = match['entity']
            counters[entity] += 1
            match['token'] = f'[MASKED_{entity}_{counters[entity]}]'

        # Replace from last to first to maintain positions
        masked_text = text
        mapping = {}
        for match in sorted(matches, key=lambda x: x['start'], reverse=True):
            start = match['start']
            end = match['end']
            token = match['token']
            original = match['text']
            masked_text = masked_text[:start] + token + masked_text[end:]
            mapping[token] = original

        return masked_text, mapping

    def demask(self, masked_text: str, mapping: Dict[str, str]) -> str:
        for token, value in mapping.items():
            masked_text = masked_text.replace(token, value)
        return masked_text

# Example usage
if __name__ == "__main__":
    masker = PIIMasker()
    
    sample_text = """
    John Doe's contact:
    Email: john.doe@example.com
    Phone: +919876543210
    DOB: 31/12/1990
    Aadhar: 1234 5678 9012
    Card: 4111-1111-1111-1111
    CVV: 123
    Expiry: 12/25
    """
    
    masked, mapping = masker.mask(sample_text)
    print("Masked Text:\n", masked)
    
    demasked = masker.demask(masked, mapping)
    print("\nDemasked Text:\n", demasked)