import { FC, useState } from 'react';
import { useParams } from 'react-router-dom';
import { Tooltip } from 'react-tooltip';

import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import { useComputeModelPrediction, useModelInformations } from '../core/api';
import { useAppContext } from '../core/context';
import { DisplayTrainingProcesses } from './DisplayTrainingProcesses';
import { ImportPredictionDataset } from './forms/ImportPredictionDataset';

export const ModelPredict: FC = () => {
  const { projectName: projectSlug } = useParams();

  const {
    appContext: { currentScheme, currentProject: project, isComputing },
  } = useAppContext();
  const [batchSize, setBatchSize] = useState<number>(32);
  // const { stopTraining } = useStopTrainBertModel(projectSlug || null);

  // available labels from context
  const [currentModel, setCurrentModel] = useState<string | null>(null);
  const { model } = useModelInformations(projectSlug || null, currentModel || null, isComputing);

  const availablePrediction =
    currentScheme &&
    currentModel &&
    project?.languagemodels.available[currentScheme] &&
    project?.languagemodels.available[currentScheme][currentModel]
      ? project?.languagemodels.available[currentScheme][currentModel]['predicted']
      : false;

  // available models
  const availableModels =
    currentScheme && project?.languagemodels.available[currentScheme]
      ? Object.keys(project?.languagemodels.available[currentScheme])
      : [];

  // compute model preduction
  const { computeModelPrediction } = useComputeModelPrediction(projectSlug || null, batchSize);

  return (
    <div className="container-fluid">
      <div className="row">
        <div className="col-8">
          <label htmlFor="selected-model">Existing models</label>
          <div className="d-flex align-items-center">
            <select
              id="selected-model"
              className="form-select"
              onChange={(e) => setCurrentModel(e.target.value)}
            >
              <option></option>
              {availableModels.map((e) => (
                <option key={e}>{e}</option>
              ))}
            </select>
          </div>

          <DisplayTrainingProcesses
            projectSlug={projectSlug || null}
            processes={project?.languagemodels.training}
            processStatus="predicting"
          />
          <div className="d-flex align-items-center">
            <label>Batch size</label>
            <a className="batch">
              <HiOutlineQuestionMarkCircle />
            </a>
            <Tooltip anchorSelect=".batch" place="top">
              Batch used for predict. Keep it small (16 or 32) for small GPU.
            </Tooltip>
            <input
              type="number"
              step="1"
              className="m-2 form-control w-25"
              value={batchSize}
              onChange={(e) => setBatchSize(Number(e.target.value))}
            />
          </div>
          {/* {isComputing && (
            <div>
              <button
                key="stop"
                className="btn btn-primary mt-3 d-flex align-items-center"
                onClick={stopTraining}
              >
                <PulseLoader color={'white'} /> Stop current process
              </button>
            </div>
          )} */}

          {currentModel && currentScheme && (
            <div>
              {model && (
                <div>
                  {availablePrediction ? (
                    <div className="alert alert-success m-4">
                      Prediction computed for this model, you can export it
                    </div>
                  ) : isComputing ? (
                    <div></div>
                  ) : (
                    <button
                      className="btn btn-info my-4  col-6"
                      onClick={() => computeModelPrediction(currentModel, 'all', currentScheme)}
                    >
                      Launch prediction complete dataset
                    </button>
                  )}
                </div>
              )}
              {model && (
                <div>
                  <ImportPredictionDataset
                    projectSlug={projectSlug || ''}
                    modelName={currentModel}
                    scheme={currentScheme}
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
