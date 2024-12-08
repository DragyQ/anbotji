import json
from typing import List, Dict, Any

class PersonalityAssessmentParser:
    def __init__(self, json_file_path: str):
        """
        Initialize the parser with a JSON file containing personality assessment questions
        
        :param json_file_path: Path to the JSON file with personality questions
        """
        with open(json_file_path, 'r') as file:
            self.data = json.load(file)
        
    def get_instructions(self) -> List[str]:
        """
        Retrieve the assessment instructions
        
        :return: List of instruction strings
        """
        return self.data.get('instructions', {}).get('general_guidance', [])
    
    def get_questions(self) -> List[Dict[str, Any]]:
        """
        Retrieve all questions
        
        :return: List of question dictionaries
        """
        return self.data.get('questions', [])
    
    def get_questions_by_test_type(self, test_type: str) -> List[Dict[str, Any]]:
        """
        Retrieve questions filtered by a specific test type
        
        :param test_type: Type of test to filter by (e.g., 'Extraversion', 'Conscientiousness')
        :return: List of questions matching the test type
        """
        return [
            question for question in self.get_questions() 
            if test_type in question.get('tests', [])
        ]
    
    def count_test_types(self) -> Dict[str, int]:
        """
        Count the occurrences of each test type across all questions
        
        :return: Dictionary with test types and their frequencies
        """
        test_counts = {}
        for question in self.get_questions():
            for test_type in question.get('tests', []):
                test_counts[test_type] = test_counts.get(test_type, 0) + 1
        return test_counts
    
    def get_test_types(self) -> List[str]:
        """
        Get a list of each test type across all questions
        
        :return: List of all unique test types
        """
        test_types = []
        for question in self.get_questions():
            for test_type in question.get('tests', []):
                if test_type not in test_types:
                    test_types.append(test_type)
        return test_types
    
    def generate_assessment_report(self) -> str:
        """
        Generate a summary report of the assessment questions
        
        :return: Formatted report string
        """
        total_questions = len(self.get_questions())
        test_counts = self.count_test_types()
        
        report = "Personality Assessment Question Analysis\n"
        report += "=====================================\n\n"
        report += f"Total Questions: {total_questions}\n\n"
        report += "Test Type Distribution:\n"
        for test_type, count in sorted(test_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_questions) * 100
            report += f"{test_type}: {count} questions ({percentage:.2f}%)\n"
        
        return report

def main():
    # Example usage
    try:
        parser = PersonalityAssessmentParser('questions.json')
        
        # Print instructions
        print("Assessment Instructions:")
        for instruction in parser.get_instructions():
            print(f"- {instruction}")
        print("\n")
        
        # Print test type distribution
        print(parser.generate_assessment_report())
        
        # Example of getting questions by test type
        print("\nAll Questions:")
        for q in parser.get_questions():
            print(f"- {q['question']}")
    
    except FileNotFoundError:
        print("JSON file not found. Please ensure 'questions.json' exists.")
    except json.JSONDecodeError:
        print("Error decoding JSON. Please check the file format.")

if __name__ == "__main__":
    main()