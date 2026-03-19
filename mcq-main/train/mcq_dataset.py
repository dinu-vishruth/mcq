# train/mcq_dataset.py
"""
Curated MCQ training dataset for fine-tuning Flan-T5.

Each example has:
- input: instruction + context passage + difficulty level
- output: structured MCQ in a parseable format

This dataset covers various subjects and all 3 difficulty levels.
"""

TRAINING_DATA = [
    # ===== EASY — Factual recall, definitions =====
    {
        "input": "Generate an easy multiple choice question from this text:\n\nPython is a high-level, interpreted programming language created by Guido van Rossum. It was first released in 1991. Python emphasizes code readability with its notable use of significant whitespace.",
        "output": "Question: Who created the Python programming language?\nA) James Gosling\nB) Guido van Rossum\nC) Dennis Ritchie\nD) Bjarne Stroustrup\nAnswer: B\nExplanation: Python was created by Guido van Rossum and first released in 1991."
    },
    {
        "input": "Generate an easy multiple choice question from this text:\n\nThe mitochondria are membrane-bound organelles found in the cytoplasm of eukaryotic cells. They generate most of the cell's supply of adenosine triphosphate (ATP), used as a source of chemical energy.",
        "output": "Question: What is the main function of mitochondria?\nA) Protein synthesis\nB) DNA replication\nC) Generating ATP for energy\nD) Cell division\nAnswer: C\nExplanation: Mitochondria generate most of the cell's ATP, which is used as chemical energy."
    },
    {
        "input": "Generate an easy multiple choice question from this text:\n\nThe Great Wall of China is a series of fortifications made of stone, brick, and other materials. It was built along the historical northern borders of China to protect against various nomadic groups. The wall spans approximately 21,196 kilometers.",
        "output": "Question: What is the approximate length of the Great Wall of China?\nA) 5,000 kilometers\nB) 10,500 kilometers\nC) 21,196 kilometers\nD) 35,000 kilometers\nAnswer: C\nExplanation: The Great Wall of China spans approximately 21,196 kilometers."
    },
    {
        "input": "Generate an easy multiple choice question from this text:\n\nPhotosynthesis is the process by which green plants convert sunlight into chemical energy. During this process, plants absorb carbon dioxide and water, using sunlight to transform them into glucose and oxygen.",
        "output": "Question: What do plants absorb during photosynthesis?\nA) Oxygen and glucose\nB) Carbon dioxide and water\nC) Nitrogen and hydrogen\nD) Oxygen and nitrogen\nAnswer: B\nExplanation: During photosynthesis, plants absorb carbon dioxide and water to produce glucose and oxygen."
    },
    {
        "input": "Generate an easy multiple choice question from this text:\n\nThe human heart has four chambers: two atria and two ventricles. The right side of the heart pumps blood to the lungs, while the left side pumps blood to the rest of the body.",
        "output": "Question: How many chambers does the human heart have?\nA) Two\nB) Three\nC) Four\nD) Five\nAnswer: C\nExplanation: The human heart has four chambers: two atria and two ventricles."
    },
    {
        "input": "Generate an easy multiple choice question from this text:\n\nThe World Wide Web was invented by Tim Berners-Lee in 1989 while working at CERN. He proposed an information management system that used hypertext to link documents across computer networks.",
        "output": "Question: Who invented the World Wide Web?\nA) Steve Jobs\nB) Bill Gates\nC) Tim Berners-Lee\nD) Mark Zuckerberg\nAnswer: C\nExplanation: Tim Berners-Lee invented the World Wide Web in 1989 while at CERN."
    },
    {
        "input": "Generate an easy multiple choice question from this text:\n\nDNA stands for deoxyribonucleic acid. It is a molecule that carries the genetic instructions used in the growth, development, functioning, and reproduction of all known living organisms.",
        "output": "Question: What does DNA stand for?\nA) Deoxyribonucleic acid\nB) Dynamic nucleic acid\nC) Diribonucleic acid\nD) Deoxyribosomal acid\nAnswer: A\nExplanation: DNA stands for deoxyribonucleic acid, a molecule carrying genetic instructions."
    },
    {
        "input": "Generate an easy multiple choice question from this text:\n\nThe Amazon River is the largest river in the world by volume of water. It flows through South America, primarily through Brazil, and is approximately 6,400 kilometers long.",
        "output": "Question: Through which continent does the Amazon River primarily flow?\nA) North America\nB) Africa\nC) South America\nD) Asia\nAnswer: C\nExplanation: The Amazon River flows through South America, primarily through Brazil."
    },
    {
        "input": "Generate an easy multiple choice question from this text:\n\nInsulin is a hormone produced by the pancreas. It regulates blood sugar levels by allowing cells to absorb glucose from the bloodstream. Diabetes occurs when the body cannot produce enough insulin or cannot use it effectively.",
        "output": "Question: Which organ produces insulin?\nA) Liver\nB) Kidney\nC) Pancreas\nD) Stomach\nAnswer: C\nExplanation: Insulin is a hormone produced by the pancreas to regulate blood sugar levels."
    },
    {
        "input": "Generate an easy multiple choice question from this text:\n\nNewton's first law of motion states that an object at rest stays at rest and an object in motion stays in motion with the same speed and in the same direction unless acted upon by an unbalanced force.",
        "output": "Question: What does Newton's first law of motion describe?\nA) The relationship between force and acceleration\nB) The conservation of energy\nC) An object's tendency to resist changes in motion\nD) The gravitational attraction between objects\nAnswer: C\nExplanation: Newton's first law describes inertia - an object's tendency to resist changes in its state of motion."
    },

    # ===== MEDIUM — Understanding, application =====
    {
        "input": "Generate a medium multiple choice question from this text:\n\nMachine learning algorithms can be categorized into supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled data to train models, unsupervised learning finds patterns in unlabeled data, and reinforcement learning learns through trial and error with rewards.",
        "output": "Question: Which type of machine learning would be most appropriate for categorizing emails as spam or not spam using a dataset of pre-labeled emails?\nA) Unsupervised learning\nB) Supervised learning\nC) Reinforcement learning\nD) Transfer learning\nAnswer: B\nExplanation: Spam classification with labeled emails is a supervised learning task because the training data already has labels (spam/not spam)."
    },
    {
        "input": "Generate a medium multiple choice question from this text:\n\nOsmosis is the movement of water molecules through a semipermeable membrane from an area of lower solute concentration to an area of higher solute concentration. This process is crucial for maintaining cell homeostasis.",
        "output": "Question: If a red blood cell is placed in a hypertonic solution (higher solute concentration outside), what would happen?\nA) The cell would swell and burst\nB) The cell would shrink as water moves out\nC) Nothing would change\nD) The cell membrane would dissolve\nAnswer: B\nExplanation: Through osmosis, water moves from low solute concentration (inside the cell) to high solute concentration (outside), causing the cell to shrink."
    },
    {
        "input": "Generate a medium multiple choice question from this text:\n\nThe greenhouse effect occurs when certain gases in Earth's atmosphere trap heat from the sun. Carbon dioxide, methane, and water vapor are the primary greenhouse gases. Without the natural greenhouse effect, Earth's average temperature would be about -18°C instead of the current 15°C.",
        "output": "Question: Why is some level of greenhouse effect considered beneficial for life on Earth?\nA) It increases oxygen levels in the atmosphere\nB) It keeps Earth's temperature warm enough to support life\nC) It blocks harmful ultraviolet radiation\nD) It creates the ozone layer\nAnswer: B\nExplanation: Without the natural greenhouse effect, Earth's temperature would be -18°C, far too cold for most life forms. The effect maintains the current average of 15°C."
    },
    {
        "input": "Generate a medium multiple choice question from this text:\n\nSQL (Structured Query Language) is used to manage relational databases. The SELECT statement retrieves data, INSERT adds new records, UPDATE modifies existing records, and DELETE removes records. JOIN operations combine rows from two or more tables based on related columns.",
        "output": "Question: Which SQL operation would you use to combine customer information from a 'customers' table with their orders from an 'orders' table?\nA) INSERT\nB) UPDATE\nC) JOIN\nD) DELETE\nAnswer: C\nExplanation: JOIN operations combine rows from two or more tables based on related columns, making it the right choice for combining customer and order data."
    },
    {
        "input": "Generate a medium multiple choice question from this text:\n\nThe water cycle describes how water moves through the environment. Evaporation turns liquid water into vapor, condensation forms clouds from water vapor, precipitation returns water to Earth's surface, and collection gathers water in oceans, rivers, and underground reserves.",
        "output": "Question: What must happen before precipitation can occur in the water cycle?\nA) Collection in rivers and oceans\nB) Evaporation from oceans\nC) Condensation of water vapor into clouds\nD) Infiltration into the ground\nAnswer: C\nExplanation: Precipitation requires water vapor to first condense into water droplets in clouds before it can fall as rain, snow, or other forms."
    },
    {
        "input": "Generate a medium multiple choice question from this text:\n\nEncryption transforms readable data into an unreadable format using an algorithm and a key. Symmetric encryption uses the same key for both encryption and decryption, while asymmetric encryption uses a public key for encryption and a private key for decryption.",
        "output": "Question: Why is asymmetric encryption preferred for secure communication between strangers over the internet?\nA) It is faster than symmetric encryption\nB) It uses shorter keys\nC) The public key can be shared openly without compromising security\nD) It requires less computing power\nAnswer: C\nExplanation: Asymmetric encryption allows the public key to be shared openly for anyone to encrypt messages, while only the private key holder can decrypt them, making it ideal for communication between parties who haven't exchanged keys before."
    },
    {
        "input": "Generate a medium multiple choice question from this text:\n\nNatural selection is the mechanism of evolution described by Charles Darwin. Organisms with traits better suited to their environment are more likely to survive and reproduce. Over many generations, these advantageous traits become more common in the population.",
        "output": "Question: How does natural selection lead to changes in a population over time?\nA) By randomly mutating all organisms equally\nB) By ensuring only the strongest individuals survive\nC) By increasing the frequency of traits that improve survival and reproduction\nD) By allowing organisms to intentionally develop beneficial traits\nAnswer: C\nExplanation: Natural selection works by making advantageous traits more common over generations because organisms with those traits are more likely to survive and reproduce."
    },
    {
        "input": "Generate a medium multiple choice question from this text:\n\nA stack is a linear data structure that follows the Last In, First Out (LIFO) principle. The push operation adds an element to the top, and the pop operation removes the top element. Stacks are used in function call management, undo operations, and expression evaluation.",
        "output": "Question: If you push elements 1, 2, 3 onto a stack and then pop twice, what element is now at the top?\nA) 3\nB) 2\nC) 1\nD) The stack is empty\nAnswer: C\nExplanation: Push 1 (bottom), push 2, push 3 (top). Pop removes 3, pop removes 2. Element 1 remains at the top."
    },
    {
        "input": "Generate a medium multiple choice question from this text:\n\nThe pH scale measures how acidic or basic a solution is, ranging from 0 to 14. A pH of 7 is neutral. Values below 7 indicate acidity, and values above 7 indicate alkalinity. Each unit change represents a tenfold difference in hydrogen ion concentration.",
        "output": "Question: A solution with pH 3 is how many times more acidic than a solution with pH 5?\nA) 2 times\nB) 10 times\nC) 100 times\nD) 1000 times\nAnswer: C\nExplanation: Each pH unit represents a tenfold difference, so 2 units apart means 10 × 10 = 100 times more acidic."
    },
    {
        "input": "Generate a medium multiple choice question from this text:\n\nObject-oriented programming organizes code around objects that combine data and behavior. The four pillars are encapsulation (hiding internal state), inheritance (creating new classes from existing ones), polymorphism (different objects responding to the same method differently), and abstraction (simplifying complex systems).",
        "output": "Question: Which OOP concept allows a Dog class and a Cat class to both have a speak() method that produces different outputs?\nA) Encapsulation\nB) Inheritance\nC) Polymorphism\nD) Abstraction\nAnswer: C\nExplanation: Polymorphism allows different objects (Dog and Cat) to respond to the same method (speak()) in different ways."
    },

    # ===== HARD — Analysis, critical thinking =====
    {
        "input": "Generate a hard multiple choice question from this text:\n\nThe CAP theorem states that a distributed data system can only simultaneously provide two of three guarantees: Consistency (every read receives the most recent write), Availability (every request receives a response), and Partition tolerance (the system continues to operate despite network partitions).",
        "output": "Question: A company needs a distributed database for a banking system where incorrect balances could cause financial loss, but occasional brief downtime during network issues is acceptable. Which CAP trade-off should they prioritize?\nA) AP - sacrifice consistency for availability and partition tolerance\nB) CP - sacrifice availability for consistency and partition tolerance\nC) CA - sacrifice partition tolerance for consistency and availability\nD) All three can be achieved with proper engineering\nAnswer: B\nExplanation: Banking requires strong consistency (correct balances). CP systems sacrifice availability during partitions, meaning brief downtime but never incorrect data. AP would risk showing wrong balances."
    },
    {
        "input": "Generate a hard multiple choice question from this text:\n\nIn economics, inflation is the general increase in prices over time. The Phillips Curve suggests an inverse relationship between unemployment and inflation. However, stagflation—when both high inflation and high unemployment occur simultaneously—challenged this model in the 1970s.",
        "output": "Question: The occurrence of stagflation in the 1970s most directly undermined which economic assumption?\nA) That monetary policy can affect interest rates\nB) That there is a stable trade-off between inflation and unemployment\nC) That supply and demand determine market prices\nD) That government spending stimulates economic growth\nAnswer: B\nExplanation: Stagflation showed that high inflation and high unemployment can coexist, contradicting the Phillips Curve's assumption of a stable inverse relationship between the two."
    },
    {
        "input": "Generate a hard multiple choice question from this text:\n\nCRISPR-Cas9 is a genome editing tool that allows scientists to alter DNA sequences. The guide RNA directs the Cas9 enzyme to a specific location in the genome, where it cuts both strands of DNA. The cell's repair mechanisms then either disable the gene or insert new genetic material.",
        "output": "Question: What is the most significant ethical concern that distinguishes germline CRISPR editing from somatic cell editing?\nA) Germline editing is technically more difficult\nB) Germline edits are passed to future generations who cannot consent\nC) Somatic cell editing is more expensive\nD) Germline editing only works in plants\nAnswer: B\nExplanation: The key ethical distinction is that germline edits affect reproductive cells and are inherited by all future descendants, who have no ability to consent to those genetic modifications."
    },
    {
        "input": "Generate a hard multiple choice question from this text:\n\nTime complexity in algorithms is measured using Big O notation. Linear search has O(n) complexity, binary search has O(log n), bubble sort has O(n²), and merge sort has O(n log n). The choice of algorithm significantly impacts performance as data size grows.",
        "output": "Question: A developer needs to search through 1 billion sorted records. Using binary search instead of linear search would reduce the maximum comparisons from 1 billion to approximately how many?\nA) 500 million\nB) 1 million\nC) 30\nD) 100,000\nAnswer: C\nExplanation: Binary search on 1 billion items needs at most log₂(10⁹) ≈ 30 comparisons, compared to 1 billion for linear search. This demonstrates the dramatic difference between O(n) and O(log n)."
    },
    {
        "input": "Generate a hard multiple choice question from this text:\n\nQuantum computing uses qubits that can exist in superposition—simultaneously 0 and 1. Quantum entanglement links qubits so measuring one instantly determines the state of its partner. These properties allow quantum computers to explore multiple solutions simultaneously, potentially solving certain problems exponentially faster than classical computers.",
        "output": "Question: Why are quantum computers unlikely to replace classical computers for all tasks despite their theoretical advantage?\nA) Quantum computers are too expensive to manufacture\nB) Quantum advantages only apply to specific problem types like factoring and optimization, not general computing\nC) Classical computers are faster at all tasks when using parallel processing\nD) Quantum computers cannot perform basic arithmetic\nAnswer: B\nExplanation: Quantum speedup is not universal. It provides exponential advantage only for specific problems (e.g., Shor's algorithm for factoring, Grover's for search). For many everyday tasks, classical computers remain equally or more efficient."
    },
    {
        "input": "Generate a hard multiple choice question from this text:\n\nNeural networks learn by adjusting weights through backpropagation. The gradient descent optimizer updates weights to minimize the loss function. However, deep networks can suffer from vanishing gradients, where gradients become extremely small in early layers, preventing effective learning.",
        "output": "Question: Residual networks (ResNets) introduce skip connections that add the input of a layer directly to its output. How does this architectural change primarily address the vanishing gradient problem?\nA) It reduces the total number of parameters in the network\nB) It provides alternative gradient pathways that bypass problematic layers\nC) It increases the learning rate automatically\nD) It eliminates the need for backpropagation entirely\nAnswer: B\nExplanation: Skip connections create shortcut paths for gradients to flow through during backpropagation, bypassing layers where gradients might vanish. This allows effective training of very deep networks."
    },
    {
        "input": "Generate a hard multiple choice question from this text:\n\nThe immune system has two components: innate immunity provides immediate, non-specific defense through barriers and inflammation, while adaptive immunity develops specific responses through B cells (producing antibodies) and T cells. Memory cells from adaptive immunity provide long-term protection against previously encountered pathogens.",
        "output": "Question: Why do vaccines typically require weeks to provide full protection, yet the same vaccine provides rapid protection upon subsequent exposure to the actual pathogen?\nA) The vaccine weakens over time and needs to be refreshed\nB) Initial exposure creates memory cells; subsequent exposure triggers a faster, stronger secondary immune response\nC) The immune system needs time to produce white blood cells\nD) Vaccines only activate innate immunity initially\nAnswer: B\nExplanation: The first vaccine exposure triggers primary adaptive immunity which is slow. But it creates memory B and T cells. When the actual pathogen appears, these memory cells mount a rapid secondary response."
    },
    {
        "input": "Generate a hard multiple choice question from this text:\n\nSupply and demand in microeconomics determines market equilibrium price. When demand increases while supply remains constant, prices rise. Price elasticity measures how sensitive quantity demanded is to price changes. Necessities like insulin tend to have inelastic demand, while luxury goods tend to have elastic demand.",
        "output": "Question: If the government imposes a price ceiling below the equilibrium price for an inelastic-demand product like insulin, what is the most likely economic consequence?\nA) Demand for insulin will decrease significantly\nB) Suppliers will increase production to meet new demand\nC) A shortage will develop because quantity demanded exceeds quantity supplied\nD) The market will quickly reach a new, lower equilibrium\nAnswer: C\nExplanation: Price ceilings below equilibrium create shortages. With inelastic demand, consumers still need insulin regardless of price, so demand stays high while the artificially low price reduces suppliers' incentive to produce."
    },
    {
        "input": "Generate a hard multiple choice question from this text:\n\nDatabase normalization organizes data to reduce redundancy. First Normal Form (1NF) eliminates repeating groups. Second Normal Form (2NF) removes partial dependencies on composite keys. Third Normal Form (3NF) removes transitive dependencies where non-key attributes depend on other non-key attributes.",
        "output": "Question: A table has columns (StudentID, CourseID, StudentName, CourseName, InstructorName, InstructorOffice) with a composite primary key of (StudentID, CourseID). If InstructorOffice depends on InstructorName which depends on CourseID, which normalization issue exists?\nA) The table is not in 1NF due to repeating groups\nB) There is a partial dependency violating 2NF only\nC) There are both partial dependencies (2NF) and transitive dependencies (3NF)\nD) The table is fully normalized\nAnswer: C\nExplanation: StudentName depends only on StudentID (partial dependency, violating 2NF). InstructorOffice depends on InstructorName, not directly on the primary key (transitive dependency, violating 3NF)."
    },
    {
        "input": "Generate a hard multiple choice question from this text:\n\nTCP provides reliable, ordered delivery of data over IP networks using three-way handshake for connection establishment, sequence numbers for ordering, and acknowledgments for reliability. UDP is connectionless and provides faster but unreliable transmission without guaranteeing delivery order.",
        "output": "Question: When designing a real-time multiplayer game, why might a developer choose UDP over TCP despite the risk of packet loss?\nA) UDP is more secure than TCP\nB) UDP has lower latency since it skips connection overhead, and stale game state data is less useful than current data\nC) UDP uses less bandwidth in all scenarios\nD) TCP cannot handle multiple concurrent connections\nAnswer: B\nExplanation: In real-time games, latency matters more than reliability. TCP's retransmission of lost packets adds delay, and by the time a lost game state packet is resent, newer state data has already made it obsolete."
    },
]


def get_training_data():
    """Return the full training dataset."""
    return TRAINING_DATA


def get_dataset_stats():
    """Print dataset statistics."""
    easy = [d for d in TRAINING_DATA if "easy" in d["input"].lower()]
    medium = [d for d in TRAINING_DATA if "medium" in d["input"].lower()]
    hard = [d for d in TRAINING_DATA if "hard" in d["input"].lower()]
    print(f"Total examples: {len(TRAINING_DATA)}")
    print(f"  Easy:   {len(easy)}")
    print(f"  Medium: {len(medium)}")
    print(f"  Hard:   {len(hard)}")


if __name__ == "__main__":
    get_dataset_stats()
