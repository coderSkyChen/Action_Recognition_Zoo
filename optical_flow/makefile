INCLUDE = $(shell pkg-config --cflags opencv)
LIBS = $(shell pkg-config --libs opencv)
OBJECTS = main.o
SOURCE = main.cpp
BIN = bin
$(OBJECTS) : $(SOURCE)
	g++ -c $(SOURCE)
$(BIN):$(OBJECTS)
	g++ -o $(BIN) $(OBJECTS) -I $(INCLUDE) $(LIBS)


clean:
	rm $(OBJECTS) $(BIN)
