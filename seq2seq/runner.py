import random
import logging

import numpy as np

from seq2seq.model_utils import save_checkpoint

__author__ = 'Shyam'


def run(args, examples, trainer, criterion, evaler, train, test, test_reporter, train_reporter):
    n_epochs = args["iters"]
    logging.info("training on %d examples for %d epochs", len(examples), n_epochs)
    random.shuffle(examples)
    seen = 0
    for epoch in range(1, n_epochs + 1):
        epoch_losses = []
        random.shuffle(examples)
        for example in examples:
            # FOR MONOTONIC MODEL, x CANNOT have any alignment characters!
            ex_loss = trainer.train_on_example(example=example,
                                               criterion=criterion)
            seen += 1
            # Keep track of loss
            epoch_losses.append(ex_loss)
            if seen > 0 and seen % args["evalfreq"] == 0:
                logging.info("seen %d loss:%.3f", seen, np.average(epoch_losses[-50:]))
                best_updated, test_acc = test_reporter.report_eval(epoch=epoch, seen=seen, evaler=evaler, examples=test)
                if best_updated and args["save"]:
                    state_dict = {
                        'args': args,
                        'enc_state_dict': trainer.encoder.state_dict(),
                        'dec_state_dict': trainer.decoder.state_dict(),
                        'enc_opt_state_dict': trainer.enc_opt.state_dict(),
                        'dec_opt_state_dict': trainer.dec_opt.state_dict(),
                    }
                    save_checkpoint(state=state_dict, is_best=True, filename=args["save"])
            if seen > 0 and seen % args["logfreq"] == 0:
                logging.info("seen %d loss:%.3f", seen, np.average(epoch_losses[-50:]))
        logging.info("epoch loss %.3f", np.average(epoch_losses))
    if args["save"]:
        logging.info("saving final model ...")
        state_dict = {
            'args': args,
            'enc_state_dict': trainer.encoder.state_dict(),
            'dec_state_dict': trainer.decoder.state_dict(),
            'enc_opt_state_dict': trainer.enc_opt.state_dict(),
            'dec_opt_state_dict': trainer.dec_opt.state_dict(),
        }
        save_checkpoint(state=state_dict, is_best=False, filename=args["save"])
    logging.info(20 * "-" + "TEST" + 20 * "-")
    test_reporter.report_eval(epoch=n_epochs, seen=seen, evaler=evaler, examples=test)
