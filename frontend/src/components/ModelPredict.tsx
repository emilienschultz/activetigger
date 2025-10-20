import { FC, useState } from 'react';
import { useParams } from 'react-router-dom';

import {
  useComputeBertModelPrediction,
  useGetPredictionsFile,
  useModelInformations,
} from '../core/api';
import { useAppContext } from '../core/context';
import { DisplayTrainingProcesses } from './DisplayTrainingProcesses';
import { ImportPredictionDataset } from './forms/ImportPredictionDataset';

export const ModelPredict: FC<{ currentModel: string | null; batchSize?: number }> = ({
  currentModel,
  batchSize,
}) => {
  const { projectName: projectSlug } = useParams();

  const useBatchSize = batchSize || 32;

  const {
    appContext: { currentScheme, currentProject: project, isComputing },
  } = useAppContext();

  // available labels from context
  const { model } = useModelInformations(
    projectSlug || null,
    currentModel || null,
    'bert',
    isComputing,
  );
  const { getPredictionsFile } = useGetPredictionsFile(projectSlug || null);

  const availablePrediction =
    currentScheme &&
    currentModel &&
    project?.languagemodels.available[currentScheme] &&
    project?.languagemodels.available[currentScheme][currentModel]
      ? project?.languagemodels.available[currentScheme][currentModel]['predicted']
      : false;

  // compute model preduction
  const { computeBertModelPrediction } = useComputeBertModelPrediction(
    projectSlug || null,
    useBatchSize,
  );

  // display external form
  const [displayExternalForm, setDisplayExternalForm] = useState<boolean>(false);
  const availablePredictionExternal =
    (currentScheme &&
      currentModel &&
      project?.languagemodels?.available?.[currentScheme]?.[currentModel]?.[
        'predicted_external'
      ]) ??
    false;

  return (
    <div className="container-fluid">
      <div className="row">
        <div className="col-12">
          <DisplayTrainingProcesses
            projectSlug={projectSlug || null}
            processes={project?.languagemodels.training}
            processStatus="predicting"
            displayStopButton={isComputing}
          />

          {currentModel && currentScheme && (
            <div>
              {model && !availablePrediction && (
                <>
                  <button
                    className="btn btn-info m-2"
                    onClick={() => computeBertModelPrediction(currentModel, 'all', currentScheme)}
                  >
                    Prediction complete dataset
                  </button>
                  <button
                    className="btn btn-info m-2"
                    onClick={() => setDisplayExternalForm(!displayExternalForm)}
                  >
                    Prediction external dataset
                  </button>
                </>
              )}
              {model && displayExternalForm && (
                <div>
                  <ImportPredictionDataset
                    projectSlug={projectSlug || ''}
                    modelName={currentModel}
                    scheme={currentScheme}
                    availablePredictionExternal={availablePredictionExternal || false}
                  />
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
