#converts PAN12 XML conv into CSV files w labeled text windows
import xml.etree.ElementTree as ET
import pandas as pd
import os

DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

#build paths to raw data
TRAIN_XML = os.path.join(DATA_DIR,"pan12-sexual-predator-identification-training-corpus-2012-05-01","pan12-sexual-predator-identification-training-corpus-2012-05-01.xml")
TRAIN_PREDATORS = os.path.join(DATA_DIR,"pan12-sexual-predator-identification-training-corpus-2012-05-01","pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt")
TEST_XML = os.path.join(DATA_DIR,"pan12-sexual-predator-identification-test-corpus-2012-05-21","pan12-sexual-predator-identification-test-corpus-2012-05-17.xml")
TEST_GT = os.path.join(DATA_DIR,"pan12-sexual-predator-identification-test-corpus-2012-05-21","pan12-sexual-predator-identification-groundtruth-problem1.txt")

#load predator IDs from text file
#input:file
#output:set of predator IDs
def load_predators(file):
    predators = set()
    with open(file, "r") as f:
        for line in f:
            predators.add(line.strip())
    return predators

#parse XML file into list of conversations
#each conv is a list of messages with author and text(also strips whitespace)
#input:xml_file
#output:list of conversations
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    conversations = []
    for conv in root.findall("conversation"):
        messages = []
        for msg in conv.findall("message"):
            author_node = msg.find("author")
            text_node = msg.find("text")
            if author_node is None:
                continue
            author = author_node.text
            if text_node is None or text_node.text is None:
               continue
            text = text_node.text.strip()
            if text == "":
                continue
            messages.append({
                "author": author,
                "text": text
            })
        if len(messages) > 0:
            conversations.append(messages)
    return conversations

#label conversations based on predator IDs
#input:list of conversations, set of predator IDs
#output:list of labeled conversations
def label_conversations(conversations, predators):
    labeled = []
    for messages in conversations:
        label = 0
        for m in messages:
            if m["author"] in predators:
                label = 1
                break
        labeled.append({
            "messages": messages,
            "label": label
        })
    return labeled

#assign stable conversation-local speaker tags in order of appearance
#input:list of messages
#output:dictionary of author IDs to speaker tags
def _speaker_map(messages):
    """
    Assign stable conversation-local speaker tags in order of appearance.
    Example: first unique author -> SPK1, second -> SPK2, ...
    """
    mapping = {}
    for m in messages:
        a = m.get("author")
        if a is None:
            continue
        if a not in mapping:
            mapping[a] = f"SPK{len(mapping) + 1}"
    return mapping


#format messages as speaker-aware text lines
#input:list of messages, dictionary of author IDs to speaker tags
#output:string of formatted messages
def _format_messages(messages, speaker_tags):
    parts = []
    for m in messages:
        text = (m.get("text") or "").strip()
        if not text:
            continue
        author = m.get("author")
        spk = speaker_tags.get(author, "SPK?")
        parts.append(f"{spk}: {text}")
    return "\n".join(parts)


#generate text windows from labeled conversations
#input:list of labeled conversations
#output:list of text windows
def generate_windows(conversations):
    dataset = []
    for conv_id, conv in enumerate(conversations):
        messages = conv["messages"]
        texts = [m["text"] for m in messages]
        label = conv["label"]
        step = max(1, len(texts) // 10)
        speaker_tags = _speaker_map(messages)
        for i in range(step, len(texts) + 1, step):
            partial_msgs = messages[:i]
            dataset.append({
                "label": label,
                "progress": i / len(texts),
                "conv_id": conv_id,
                "text": _format_messages(partial_msgs, speaker_tags),
                "window_strategy": "full",
            })
    return dataset

#process training data
#input:none
#output:none
def process_training():
    print("Processing training data...")
    predators = load_predators(TRAIN_PREDATORS)
    conversations = parse_xml(TRAIN_XML)
    labeled = label_conversations(conversations, predators)
    windows = generate_windows(labeled)
    df = pd.DataFrame(windows)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(f"{PROCESSED_DIR}/train_windows.csv", index=False)
    print("Training data saved.")

#process test data
#input:none
#output:none
def process_test():
    print("Processing test data...")
    predators = load_predators(TEST_GT)
    conversations = parse_xml(TEST_XML)
    labeled = label_conversations(conversations, predators)
    windows = generate_windows(labeled)
    df = pd.DataFrame(windows)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(f"{PROCESSED_DIR}/test_windows.csv", index=False)
    print("Test data saved.")


def main():
    process_training()
    process_test()


if __name__ == "__main__":
    main()