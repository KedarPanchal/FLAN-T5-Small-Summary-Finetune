from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import re

model = AutoModelForSeq2SeqLM.from_pretrained("./flan-t5-small-qa")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

pipe = pipeline(
    task="summarization",
    model=model,
    tokenizer=tokenizer
)
prompt = "Leptin is an adipokine, which plays key roles in regulation of glucose metabolism and energy homeostasis. Therefore, identification of a short peptide from leptin which improves glucose-metabolism and energy-homeostasis could be of significant therapeutic importance. Mutational studies demonstrated that N-terminal of human leptin hormone is crucial for activation of leptin-receptor while its C-terminal seems to have lesser effects in it. Thus, for finding a metabolically active peptide and complimenting the mutational studies on leptin, we have identified a 17-mer (leptin-1) and a 16-mer (leptin-2) segment from its N-terminal and C-terminal, respectively. Consistent with the mutational studies, leptin-1 improved glucose-metabolism by increasing glucose-uptake, GLUT4 expression and its translocation to the plasma membrane in L6-myotubes, while leptin-2 was mostly inactive. Leptin-1-induced glucose-uptake is mediated through activation of AMPK, PI3K, and AKT proteins since inhibitors of these proteins inhibited the event. Leptin-1 activated leptin-receptor immediate downstream target protein, JAK2 reflecting its possible interaction with leptin-receptor while leptin-2 was less active. Furthermore, leptin-1 increased mitochondrial-biogenesis and ATP-production, and increased expression of PGC1α, NRF1, and Tfam proteins, that are important regulators of mitochondrial biogenesis. The results suggested that leptin-1 improved energy-homeostasis in L6-myotubes, whereas, leptin-2 showed much lesser effects. In diabetic, db/db mice, leptin-1 significantly decreased blood glucose level and improved glucose-tolerance. Leptin-1 also increased serum adiponectin and decreased serum TNF-α and IL-6 level signifying the improvement in insulin-sensitivity and decrease in insulin-resistance, respectively in db/db mice. Overall, the results show the identification of a short peptide from the N-terminal of human leptin hormone which significantly improves glucose-metabolism and energy-homeostasis."
prompt = re.sub(r"<[^<]+>", "", prompt)
print(pipe(prompt, min_new_tokens=10, do_sample=False))