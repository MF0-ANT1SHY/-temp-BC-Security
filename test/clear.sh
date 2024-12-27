# mkdir for archive with date
mkdir -p archive/$(date +%Y-%m-%d)

# move all files to archive
mv *.err *.out *.log archive/$(date +%Y-%m-%d)