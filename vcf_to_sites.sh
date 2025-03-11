
# Shell script that convernts a VCF to the sitefile used in the 2-SFS pipeline

# The script queries a VCF file, dumps relevant information to a text file, then uses the python
# script sitefile.py to convert to the sitefile that gets read by the 2-SFS pipeline

vcf_file=/path/to/vcf_file.vcf.gz

# Things you may wish to replace:
# 	FILTER -> passing all filters may be indicated with "PASS" or "."
#	REF="A"... && ALT="A"... -> Filters for SNPs, there may be a better way to do this
# 	    depending on the format of your VCF
# Everything after "-f" gets dumped to the text file: chromosome, position (on the chromosome),
#     reference allele (i.e. G,A,T,or C), alternate allele, total number of alleles,
#     alterate/minor allele count
bcftools query -i 'FILTER="." && (REF="A" || REF="G" || REF="C" || REF="T") && (ALT="A" | ALT="G" | ALT="C" | ALT="T") && AC>=1' -f '%CHROM %POS %REF %ALT %AN %AC{0}\n' -c all $vcf_file >> holder.txt

python sitefile.py

rm holder.txt
