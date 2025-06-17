import { FC, useCallback, useEffect, useState } from 'react';
import { ReactSortable } from 'react-sortablejs';
import { useAppContext } from '../core/context';
import { reorderLabels } from '../core/utils';

interface MulticlassInputProps {
  elementId: string;
  labels: string[];
  postAnnotation: (label: string, elementId: string) => void;
}

interface LabelType {
  id: number;
  label: string;
}

export const MulticlassInput: FC<MulticlassInputProps> = ({
  elementId,
  postAnnotation,
  labels,
}) => {
  // get the context and set the labels
  const {
    appContext: { displayConfig },
    setAppContext,
  } = useAppContext();

  const [availableLabels, setAvailableLabels] = useState<LabelType[]>(
    reorderLabels(labels || [], displayConfig.labelsOrder || []).map((label, index) => ({
      id: index,
      label: label,
    })),
  );

  const handleKeyboardEvents = useCallback(
    (ev: KeyboardEvent) => {
      // prevent shortkey to perturb the inputs
      const activeElement = document.activeElement;
      const isFormField =
        activeElement?.tagName === 'INPUT' ||
        activeElement?.tagName === 'TEXTAREA' ||
        activeElement?.tagName === 'SELECT';
      if (isFormField) return;

      availableLabels.forEach((item, i) => {
        if (ev.code === `Digit` + (i + 1) || ev.code === `Numpad` + (i + 1)) {
          postAnnotation(item.label, elementId);
        }
      });
    },
    [availableLabels, postAnnotation, elementId],
  );

  useEffect(() => {
    // manage keyboard shortcut if less than 10 label
    if (availableLabels.length > 0 && availableLabels.length < 10) {
      document.addEventListener('keydown', handleKeyboardEvents);
    }

    return () => {
      if (availableLabels.length > 0 && availableLabels.length < 10) {
        document.removeEventListener('keydown', handleKeyboardEvents);
      }
    };
  }, [availableLabels, handleKeyboardEvents]);

  // update the labels in the state and context
  const updateLabels = (newLabels: LabelType[]) => {
    setAvailableLabels(newLabels);
    setAppContext((state) => ({
      ...state,
      displayConfig: {
        ...state.displayConfig,
        labelsOrder: newLabels.map((e) => e.label),
      },
    }));
  };

  return (
    <ReactSortable list={availableLabels} setList={updateLabels} tag="div">
      {
        // display buttons for label from the user
        availableLabels.map((e, i) => (
          <button
            type="button"
            key={e.label}
            value={e.label}
            className="btn btn-primary grow-1 gap-2 justify-content-center m-1"
            onClick={(v) => {
              postAnnotation(v.currentTarget.value, elementId);
            }}
          >
            {e.label} <span className="badge text-bg-secondary">{i + 1}</span>
          </button>
        ))
      }
    </ReactSortable>
  );
};
