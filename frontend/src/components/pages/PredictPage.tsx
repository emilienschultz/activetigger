import { FC, useState } from 'react';
import { useParams } from 'react-router-dom';
import PulseLoader from 'react-spinners/PulseLoader';
import { Tooltip } from 'react-tooltip';

import { FaTools } from 'react-icons/fa';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import {
  useComputeModelPrediction,
  useModelInformations,
  useStopTrainBertModel,
} from '../../core/api';
import { useAppContext } from '../../core/context';
import { ImportPredictionDataset } from '../forms/ImportPredictionDataset';
import { ProjectPageLayout } from '../layout/ProjectPageLayout';

export const ProjectPredictPage: FC = () => {
  const { projectName: projectSlug } = useParams();

  const {
    appContext: { currentScheme, currentProject: project, isComputing },
  } = useAppContext();
  const [batchSize, setBatchSize] = useState<number>(32);
  const { stopTraining } = useStopTrainBertModel(projectSlug || null);

  // available labels from context
  const [currentModel, setCurrentModel] = useState<string | null>(null);
  const { model } = useModelInformations(projectSlug || null, currentModel || null, isComputing);

  const availablePrediction =
    currentScheme && currentModel && project?.languagemodels.available[currentScheme][currentModel]
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
    <ProjectPageLayout projectName={projectSlug || null} currentAction="predict">
      <div className="container-fluid">
        <div className="row">
          <div className="col-8">
            <div className="explanations">
              Extend prediction on the whole corpus
              <a className="problems m-2">
                <FaTools />
              </a>
              <Tooltip anchorSelect=".problems" place="top">
                If the model doesn't train directly, it can be due to lack of available GPU. GPU
                processes are limited, and there is a queue to access them.
              </Tooltip>
            </div>

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

            {/* Display the progress of training models */}
            {project?.languagemodels.training &&
              Object.keys(project.languagemodels.training).length > 0 && (
                <div className="mt-3">
                  Current process:
                  <ul>
                    {Object.entries(
                      project?.languagemodels.training as Record<
                        string,
                        Record<string, string | number>
                      >,
                    ).map(([_, v]) => (
                      <li key={v.name}>
                        {v.name} - {v.status} :{' '}
                        <span style={{ fontWeight: 'bold' }}>
                          {Math.round(Number(v.progress))} %
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
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
            {isComputing && (
              <div>
                <button
                  key="stop"
                  className="btn btn-primary mt-3 d-flex align-items-center"
                  onClick={stopTraining}
                >
                  <PulseLoader color={'white'} /> Stop current process
                </button>
              </div>
            )}

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
    </ProjectPageLayout>
  );
};
