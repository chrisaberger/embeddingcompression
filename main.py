import utils
import core
import quantizers
import bucketers

args = utils.parse_arguments()
vocab, embedding = utils.load_embeddings(args.filename)
"""
Based on the user input declear RowBucketer, ColBucketer, and Quantizer 
objects.
"""
if args.row_bucketer == "uniform":
    row_bucketer = bucketers.UniformRowBucketer(args.num_row_buckets)
elif args.row_bucketer == "kmeans":
    row_bucketer = bucketers.KmeansRowBucketer(args.num_row_buckets)

if args.col_bucketer == "uniform":
    col_bucketer = bucketers.UniformColBucketer(args.num_col_buckets)
elif args.col_bucketer == "kmeans":
    col_bucketer = bucketers.KmeansColBucketer(args.num_col_buckets)

if args.quantizer == "uniform_fp":
    quantizer = quantizers.FixedPointQuantizer(args.num_bits)
elif args.quantizer == "kmeans":
    quantizer = quantizers.KmeansQuantizer(args.num_centroids)
elif args.quantizer == "uniform_mt":
    quantizer = quantizers.MidtreadQuantizer(args.num_bits)
elif args.quantizer == "lloydmax":
    quantizer = quantizers.LloydMaxQuantizer(args.num_centroids, args.quant_num_rows, args.quant_num_cols)
"""
Run the bucketing algorithms! The bucketing algorithms are always run by running
the 'row_bucketer' first then by running the 'col_bucketer' second. If you must
run in the other order take the transpose of the embedding before this method.
TODO add code that takes the transpose in the 'core.finish' method. 
"""
buckets, row_reorder, col_reorder = core.bucket(row_bucketer, col_bucketer,
                                                embedding)

# Run the quantization scheme inside of each bucket.
q_buckets, num_bytes = core.quantize(buckets, quantizer)
"""
Extra bytes are needed when columns are reordered. We are free to reorder rows
but when we reorder columns we must maintain information to get us back to the 
original order. 
"""
num_bytes += row_bucketer.extra_bytes_needed()
num_bytes += col_bucketer.extra_bytes_needed()
filename = utils.create_filename(args.output_folder, args.filename,
                                 row_bucketer, col_bucketer, quantizer,
                                 num_bytes)
print("Output filename: " + filename)
core.finish(q_buckets, num_bytes, embedding, vocab, row_reorder, col_reorder,
            filename)
