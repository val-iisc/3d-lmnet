import tensorflow as tf

def get_summary(ph_summary):
	'''
	Takes a list of placeholder for tensorboard summary
	and returns merged_summ variable in tensorflow
	Args:
		ph_summary: list of placeholders used for logging
	'''
	summary_loss = []
	for ph in ph_summary:
		summary_loss.append(tf.summary.scalar(ph.name, ph))
	merged_summ = tf.summary.merge(summary_loss)
	return merged_summ

def reset_stats(ph_summary, stats):
	'''
	Reinitializes dictionary losses to zero after every
	PRINT_N iterations and beginning of epoch
	Args:
		ph_summary: list of placeholders used for logging
		stats: dictionary mapping from ph to value
	'''
	stats = {}
	for ph in ph_summary:
		stats[ph] = 0.
	return stats

def update_stats(ph_summary, _summary_losses, stats, step):
	'''
	Update stats dictionary after every iteration
	Args:
		ph_summary: list of placeholders used for logging
		_summary_losses: list of values of losses corresponding to
				 values in list ph_summary respectively 
				 for one batch
		stats: dictionary mapping from ph to value
		step:  no. of iterations to average the losses over
	'''
	for ph, _loss in zip(ph_summary, _summary_losses):
		stats[ph] += _loss / step
	return stats
