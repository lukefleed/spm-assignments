# Compilatore C++ e flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -pthread -Iinclude
# Per debug:
# CXXFLAGS = -std=c++17 -Wall -Wextra -g -pthread -Iinclude

# Nome dell'eseguibile
TARGET = collatz_par

# Directory sorgenti e build
SRCDIR = src
BUILDDIR = build
OBJDIR = $(BUILDDIR)/obj
RESULTDIR = results

# Trova tutti i file sorgente .cpp
SOURCES = $(wildcard $(SRCDIR)/*.cpp)

# Crea la lista dei file oggetto .o nella directory di build
OBJECTS = $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SOURCES))

# Regola di default: compila l'eseguibile
all: $(BUILDDIR)/$(TARGET)

# Regola per creare l'eseguibile finale linkando gli oggetti
$(BUILDDIR)/$(TARGET): $(OBJECTS) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Regola per compilare i file sorgente .cpp in file oggetto .o
# Crea le directory necessarie per gli oggetti
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Crea le directory necessarie se non esistono
$(BUILDDIR):
	@mkdir -p $(BUILDDIR)
$(OBJDIR):
	@mkdir -p $(OBJDIR)
$(RESULTDIR):
	@mkdir -p $(RESULTDIR)

# Regola per pulire i file generati
clean:
	rm -rf $(BUILDDIR) $(TARGET) # Rimuove build e eseguibile root (se presente)
	# Opzionale: rimuovere anche i risultati
	# rm -f $(RESULTDIR)/*.csv

# --- Target per i Test ---

# Esegue i test di correttezza
test_correctness: $(BUILDDIR)/$(TARGET)
	@echo "--- Running Correctness Tests ---"
	./$(BUILDDIR)/$(TARGET) --test-correctness

# Esegue i test di performance e salva l'output CSV
# Assicura che la directory dei risultati esista
test_performance: $(BUILDDIR)/$(TARGET) | $(RESULTDIR)
	@echo "--- Running Performance Tests (outputting CSV to $(RESULTDIR)/performance_data.csv) ---"
	./$(BUILDDIR)/$(TARGET) --test-performance > "$(RESULTDIR)/performance_data.csv"
	@echo "--- Performance test data saved to $(RESULTDIR)/performance_data.csv ---"


# Phony targets non corrispondono a file
.PHONY: all clean test_correctness test_performance
