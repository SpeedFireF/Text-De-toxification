import torch
import torch.nn as nn
from transformers import BertTokenizer
import torch.optim as optim
from tqdm import tqdm


class ToxicWordClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, dropout_rate=0.2):
        super(ToxicWordClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)  # Apply dropout to the embedded input
        output = self.fc(embedded)
        output = self.sigmoid(output)
        return output


def build_dataset():
    # Load toxic and non-toxic words from external text files
    toxic_words = [line.strip() for line in open('/Users/damirabdulaev/Downloads/toxic_words.txt', 'r', encoding='utf-8')]
    non_toxic_words = [line.strip() for line in open('/Users/damirabdulaev/Downloads/positive-words.txt', 'r', encoding='utf-8')]
    all_words = toxic_words + non_toxic_words
    labels = [1] * len(toxic_words) + [0] * len(non_toxic_words)
    return all_words, labels


def build_tokenizer():
    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Tokenize and convert your sentences to model input
    sentences = build_dataset()[0]

    # Tokenize and convert sentences to input indices
    input_ids = []

    for sentence in sentences:
        # Tokenize the sentence and add special tokens
        encoded_dict = tokenizer(
            sentence,
            add_special_tokens=False,
            truncation=True,
            max_length=1,
            padding='max_length',
            return_tensors='pt'
        )

        # Extract the input IDs and attention mask
        input_ids.append(encoded_dict['input_ids'])

    # Convert the lists of tensors to a single tensor
    word_indices = torch.cat(input_ids, dim=0)

    return tokenizer, word_indices


def save_model(model):
    torch.save(model.state_dict(), 'twc.pth')


def load_model():
    # Initialize the model
    model = ToxicWordClassifier(vocab_size, embedding_dim, output_dim)

    # Load the saved model state_dict
    model.load_state_dict(torch.load('twc.pth'))

    return model


def train_model():
    tokenizer, word_indices = build_tokenizer()[0], build_tokenizer()[1]

    # Create a PyTorch model
    vocab_size = len(tokenizer.vocab)  # Assuming you've defined 'vocabulary'
    embedding_dim = 100  # Adjust as needed
    output_dim = 1  # Assuming binary classification

    model = ToxicWordClassifier(vocab_size, embedding_dim, output_dim)

    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert labels to tensors
    labels = build_dataset()[1]
    labels = torch.tensor(labels, dtype=torch.float, requires_grad=True)

    # Training loop
    num_epochs = 10  # Specify the number of training epochs

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = len(labels)

        # Wrap your training data with tqdm for the progress bar
        for indices, label in tqdm(zip(word_indices, labels), total=len(labels), desc=f'Epoch {epoch + 1}'):
            optimizer.zero_grad()
            inputs = torch.tensor(indices, dtype=torch.long)

            # Forward pass
            outputs = model(inputs)[0][0]

            # Calculate the loss
            loss = criterion(outputs, label)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            predicted = (outputs > 0.5).float()

            # Compute accuracy
            correct = (predicted == label).float()
            total_correct += correct.sum().item()
            total_loss += loss.item()

        # Calculate average loss and accuracy for the epoch
        avg_loss = total_loss / total_samples
        accuracy = (total_correct / total_samples) * 100.0

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    print("Training complete")

    save_model(model)

    return model


