PDFS=proposal.pdf doc.pdf

all: $(PDFS)

%.pdf: %.tex
	latexmk -pdf $<
	@kill -HUP $$(pidof mupdf)

clean:
	rm -f ${PDFS} *.{aux,log}
