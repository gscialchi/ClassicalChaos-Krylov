"""
This library provides a class to define calculation pipelines: a sequence of
calculations to do in order to get an end result. The idea is, if the wanted
end results are not already-stored data, then call its `providers` in order
to get the inputs to do the end calculation. Do this recursively until done.
"""


class Dummy:
    def __init__(self):
        ...


def get_free_args(doer):
    # dummy values don't count as arguments that are set
    set_args = [k for k, v in doer.get_args().items()
                if not isinstance(v, Dummy)]
    all_args = doer.spec.args
    return [arg for arg in all_args if arg not in set_args]


class Pipeline:
    def __init__(self, doers):
        self.doers = doers
        for i, doer in enumerate(self.doers):
            doer_copy = doer.copy()
            doer_copy.except_if_not_found = True
            self.doers[i] = doer_copy

        self.do_queue = [self.doers[-1]]
        self.results = {}


    def _who_provides(self, arg):
        for doer in self.doers:
            if arg in doer.provides:
                return doer


    def doit(self):
        iter_count = 0
        max_count = 2*len(self.doers)
        while not len(self.do_queue) == 0:
            if iter_count > max_count:
                raise Exception("""Something went wrong: number of iterations
                                should never need to exceed twice the number
                                of doers""")

            doer = self.do_queue[0]

            # if wasn't calculated already, then set dummy values for free args
            args = dict([(k, Dummy()) if k not in self.results
                         else (k, self.results[k])
                         for k in get_free_args(doer)])
            doer.set_args(dic=args)

            # check if all args were filled just now, if True can calculate
            no_free_args = len(get_free_args(doer)) == 0
            if no_free_args:
                doer.except_if_not_found = False
                print(f'{doer.alias}: inputs available, will calculate.')
            else:
                print(f'{doer.alias}: some missing inputs, will try to load.')

            try: # try to load or calculate
                if ((not doer.load) or doer.disabled) and (not no_free_args):
                    raise FileNotFoundError # won't be able to calculate

                output = doer.doit()

                # put output in results
                if len(doer.provides) == 1: # otherwise zips the output itself
                    self.results[doer.provides[0]] = output
                else:
                    for k, v in zip(doer.provides, output):
                        self.results[k] = v

                self.do_queue.remove(doer) # remove from queue if successful

            except FileNotFoundError: # if failed, queue doers to get inputs
                                      # and try again later
                print(f'{doer.alias}: missing inputs, will queue providers and try again later.')
                k = 0
                for arg in get_free_args(doer):
                    if not arg in self.results:
                        provider = self._who_provides(arg)
                        if not provider in self.do_queue:
                            self.do_queue.insert(k+1, provider)
                    k += 1
                self.do_queue.insert(k+1, doer) # move further in queue
                self.do_queue.remove(doer) # for when requirements are met
                doer.except_if_not_found = False # next time won't be dummy

            iter_count += 1
