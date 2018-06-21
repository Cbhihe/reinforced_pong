PDFS=doc.pdf slides2.pdf

all: $(PDFS)

%.pdf: %.tex
	latexmk -pdf $<
	@kill -HUP $$(pidof mupdf)

clean:
	rm -f ${PDFS} *.{aux,log}
