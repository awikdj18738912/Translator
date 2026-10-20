"""Text snapshot replay using the same window callback as the Web pipeline.

This adapter has no ASR client. Its input is an entire revisable Chinese text
snapshot, such as an upstream machine translation hypothesis.
"""
from threading import Lock

from .entity_matcher import EntityCandidateMatcher, EntityFuzzyMode
from .protection import EntityProtector
from .session_memory import SessionEntityMemory
from .window_refinement import CumulativeWindowRefinement, StreamingRefinementDisplay


class OfflineTextRefinementSession:
    def __init__(self, refine_update, definitions, *, domain='general',
                 fuzzy_mode=EntityFuzzyMode.OFF, rule_protection=True,
                 group_numeric_self_corrections=False, final_reprocess_rejected=False,
                 call_stats=None):
        self._revision = 0
        self._closed = False
        self._lock = Lock()
        self.final_reprocess_rejected = final_reprocess_rejected
        self.model_calls = []
        self._call_stats = call_stats
        self.definitions = tuple(definitions)
        self.protector = EntityProtector(self.definitions, session_memory=SessionEntityMemory(),
            enable_rule_protection=rule_protection, protect_numeric_spans=False)
        self.matcher = (EntityCandidateMatcher(self.definitions, selected_domain=domain,mode=fuzzy_mode)
                        if self.definitions and fuzzy_mode is not EntityFuzzyMode.OFF else None)

        def invoke(*args, **kwargs):
            kwargs.pop('_trace_window_chunks', None)
            kwargs.pop('_trace_required_chunks', None)
            traces = []
            if self._call_stats is not None:
                kwargs['call_stats'] = self._call_stats
            result = refine_update(*args, **kwargs, trace_segments=traces)
            self.model_calls.extend(traces)
            return result

        def pending(*args, **kwargs):
            return invoke(*args, **kwargs, pending_only=True)

        self.windows = CumulativeWindowRefinement(invoke,window_size=3,
            one_punctuation_window=True,fixed_groups=True,prepare_pending=pending,
            group_numeric_self_corrections=group_numeric_self_corrections,adjacent_context=True)
        self.display = StreamingRefinementDisplay()

    def metrics_snapshot(self):
        """Return cumulative backend metrics for this text refinement run."""
        if self._call_stats is None:
            return {'refiner_model_calls': sum(
                len(trace.get('model_calls', [])) for trace in self.model_calls
            ), 'gate_skipped_groups': 0}
        snapshot = self._call_stats.snapshot()
        return {
            'refiner_model_calls': snapshot['call_count'],
            'gate_skipped_groups': snapshot['gate_skipped_segment_count'],
            'refiner_retry_calls': snapshot['retry_call_count'],
        }

    def update(self, text: str, *, revision: int, final: bool = False,
               on_progress=None):
        if not isinstance(text,str) or len(text)>32000:
            raise ValueError('text must be a bounded Chinese snapshot')
        with self._lock:
            if self._closed:
                raise ValueError('text session already finished')
            if isinstance(revision,bool) or not isinstance(revision,int) or revision<=self._revision:
                raise ValueError('revision must increase monotonically')
            # No ASR confidence exists for translated text. Never manufacture
            # a high-confidence signal for gates trained on ASR hypotheses.
            self.windows.on_group_progress = on_progress
            try:
                result = self.windows.update(text,'Chinese',final,self.protector,None,self.matcher,
                    confidence_metadata={},reprocess_rejected=final and self.final_reprocess_rejected)
            finally:
                self.windows.on_group_progress = None
            self._revision = revision
            self._closed = final
            self.display.accept(result,revision)
            return {**result,**self.display.compose(text),'translation_revision':revision,
                    'final':final,'input_kind':'translated_text_snapshot',
                    'asr_confidence':None,'asr_confidence_metadata':{}}
