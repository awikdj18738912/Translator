"""Event-loop-owned display state for independently advancing text stages."""
from .refinement_guard import join_refined_segments
from .window_refinement import StreamingRefinementDisplay


class TranslationStreamDisplay:
    def __init__(self):
        self.translation = ""
        self.translation_revision = 0
        self.refinement_revision = 0
        self.refined = StreamingRefinementDisplay()

    def project(self, event):
        kind = event.get("event")
        if kind not in {"translation", "update", "final", "refinement_progress"}:
            return event
        revision = event["revision"]
        if kind == "translation" and revision >= self.translation_revision:
            self.translation = event["translation"]
            self.translation_revision = revision
        if kind in {"update", "final", "refinement_progress"} and revision >= self.refinement_revision:
            # An incomplete group is visible raw text, not a validated span.
            # Never accept the composed full column as a refined prefix.
            spans = event.get("refinement_source_spans", ())
            committed = []
            for span in spans:
                if span.get("state") != "committed":
                    break
                committed.append(span)
            if committed:
                self.refined.accept({
                    "raw_text": "".join(span["source_text"] for span in committed),
                    "clean_text": join_refined_segments(span["clean_text"] for span in committed),
                    "refinement_source_spans": committed,
                }, revision)
            self.refinement_revision = revision
        view = self.refined.compose(self.translation)
        if kind == "final":
            # The final result has passed the existing guard/fallback path.
            # Keep its exact aggregate cleanup, including punctuation joins.
            view.update(display_refined_text=event["clean_text"], pending_raw_text="",
                        display_text=event["clean_text"], has_pending_refinement=False,
                        display_revision=revision)
        return {**event, **view, "translation": self.translation,
                "clean_text": view["display_text"],
                "translation_revision": self.translation_revision,
                "refinement_revision": self.refinement_revision}
