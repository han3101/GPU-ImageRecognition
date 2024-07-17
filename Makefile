# Variables
CC := g++
TARGET := "ImageProcessing"
CFLAGS := -Iinclude -I/usr/include -std=c++17 -DCL_TARGET_OPENCL_VERSION=300 -fopenmp
# CFLAGS := -I/usr/local/cuda-12.3/targets/x86_64-linux/include -std=c++17 -DCL_TARGET_OPENCL_VERSION=300 -fopenmp 
LDFLAGS = -L/usr/lib/x86_64-linux-gnu -lOpenCL
BUILDDIR := bin
OUTPUT := output
SRCDIR := src
SRCEXT := cpp
SOURCES := $(wildcard $(SRCDIR)/*.$(SRCEXT))
OBJS := $(patsubst $(SRCDIR)/%, $(BUILDDIR)/%, $(SOURCES:.$(SRCEXT)=.o))

# Define additional flags for profiling
ifdef PROFILE
CFLAGS += -DPROFILE
endif

# Rule to build source files
$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@printf "Building..\n";
	@mkdir -p $(BUILDDIR)
	@echo "  $(notdir $@) from $(notdir $<)"
	@$(CC) $(CFLAGS) -c -o $@ $<

# Rule to link directories
$(TARGET): $(OBJS) | $(OUTPUT)
	@printf "Linking..\n";
	@echo "  $(notdir $(OBJS))"
	@$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Rule to create OUTPUT directory
$(OUTPUT):
	@printf "Making Output Dir..\n";
	@mkdir -p $(OUTPUT)


# Rule to build all
all: clean $(TARGET) $(OUTPUT)

# Rule to build with Profile
profile:
	$(MAKE) PROFILE=1

# Clean up build files
clean-all:
	@printf "\e[31m\e[1mCleaning...\e[0m\n"
	@echo "  /$(BUILDDIR)"
	@echo "  /$(OUTPUT)"
	@echo "  /$(TARGET)"
	@$(RM) -r $(BUILDDIR) $(OBJECTS)
	@$(RM) "./$(TARGET)"
	@$(RM) -r "./$(OUTPUT)"

# Run rule to execute the compiled program
r:
	@printf "Running $(TARGET)\n"
	@./$(TARGET)

# Phony targets
.PHONY: all clean-all r profile
