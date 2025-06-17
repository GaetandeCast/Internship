TEXFILE=main
OUTDIR=build
PDF=$(TEXFILE).pdf

.PHONY: all clean

all:
	mkdir -p $(OUTDIR)
	latexmk -pdf -interaction=nonstopmode -synctex=1 -outdir=$(OUTDIR) $(TEXFILE).tex
	cp $(OUTDIR)/$(PDF) ./$(PDF)

clean:
	latexmk -C -outdir=$(OUTDIR) $(TEXFILE).tex
