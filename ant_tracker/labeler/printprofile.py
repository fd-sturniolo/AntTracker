import pstats

p = pstats.Stats('profile')
p.sort_stats('cumtime').print_stats("encoder",10)
p.sort_stats('cumtime').print_stats("classes",10)
# p.strip_dirs().sort_stats('cumulative').print_stats(10)
p.strip_dirs().sort_stats('time').print_stats(10)
