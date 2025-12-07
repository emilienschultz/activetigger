import { FC, useCallback, useEffect, useState } from 'react';
import { ReactSortable } from 'react-sortablejs';
import { Tooltip } from 'react-tooltip';
import { useAppContext } from '../core/context';
import { reorderLabels } from '../core/utils';
import { ElementOutModel } from '../types';

interface MulticlassInputProps {
  elementId: string;
  labels: string[];
  postAnnotation: (label: string, elementId: string) => void;
  small?: boolean;
  phase?: string;
  element?: ElementOutModel;
}

interface LabelType {
  id: number;
  label: string;
}

export const MulticlassInput: FC<MulticlassInputProps> = ({
  elementId,
  postAnnotation,
  labels,
  phase,
  element,
  small = false,
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

      // NEW ACTION FOR KEY "P"
      if (ev.code === 'KeyP') {
        postAnnotation(element?.predict.label || '', elementId);
      }

      availableLabels.forEach((item, i) => {
        if (ev.code === `Digit` + (i + 1) || ev.code === `Numpad` + (i + 1)) {
          postAnnotation(item.label, elementId);
        }
      });
    },
    [availableLabels, postAnnotation, elementId, element],
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

  const predict_proba = element?.predict.proba ? element.predict.proba.toFixed(2) : 'NA';
  const predict_entropy = element?.predict.entropy ? element.predict.entropy.toFixed(2) : 'NA';

  return (
    <>
      {
        //display proba
        phase == 'train' && displayConfig.displayPrediction && element?.predict.label && (
          <div className="d-flex mb-2 justify-content-center display-prediction">
            <button
              type="button"
              value={element?.predict.label as unknown as string}
              className={`btn ${small ? 'btn-sm' : ''} btn-secondary grow-1 gap-2 justify-content-center m-1 elementpredicted`}
              onClick={(e) => {
                postAnnotation(e.currentTarget.value, elementId);
              }}
            >
              Predicted : {element?.predict.label} <span className="badge text-bg-primary">p</span>
            </button>
            <Tooltip anchorSelect=".elementpredicted" place="top">
              {`proba: ${predict_proba}, entropy: ${predict_entropy}`}
            </Tooltip>
          </div>
        )
      }
      <ReactSortable list={availableLabels} setList={updateLabels} tag="div">
        {
          // display buttons for label from the user
          availableLabels.map((e, i) => (
            <button
              type="button"
              key={e.label}
              value={e.label}
              className={`btn ${small ? 'btn-sm' : ''} btn-primary grow-1 gap-2 justify-content-center m-1`}
              onClick={(v) => {
                postAnnotation(v.currentTarget.value, elementId);
              }}
            >
              {e.label} <span className="badge text-bg-secondary">{i + 1}</span>
            </button>
          ))
        }
      </ReactSortable>
    </>
  );
};
