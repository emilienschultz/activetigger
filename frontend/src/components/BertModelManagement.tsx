import { FC, useState } from 'react';
import { Tab, Tabs } from 'react-bootstrap';
import { FaTools } from 'react-icons/fa';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import { MdOutlineDeleteOutline } from 'react-icons/md';
import { Tooltip } from 'react-tooltip';
import { DisplayTrainingProcesses } from '../components/DisplayTrainingProcesses';
import { ModelCreationForm } from '../components/forms/ModelCreationForm';
import { ModelParametersTab } from '../components/ModelParametersTab';
import { LossChart } from '../components/vizualisation/lossChart';
import { ProjectStateModel } from '../types';

import { SubmitHandler, useForm } from 'react-hook-form';
import { DisplayScores } from '../components/DisplayScores';
import { useDeleteBertModel, useModelInformations, useRenameBertModel } from '../core/api';
import { useNotifications } from '../core/notifications';
import { MLStatisticsModel } from '../types';

interface renameModel {
  new_name: string;
}

interface LossData {
  epoch: { [key: string]: number };
  val_loss: { [key: string]: number };
  val_eval_loss: { [key: string]: number };
}

interface BertModelManagementProps {
  projectSlug: string | null;
  isComputing: boolean;
  project: ProjectStateModel | null;
  currentScheme: string | null;
  availableBertModels: string[];
}

export const BertModelManagement: FC<BertModelManagementProps> = ({
  projectSlug,
  currentScheme,
  project,
  availableBertModels,
  isComputing,
}) => {
  const { notify } = useNotifications();

  // current model and automatic selection

  const [currentBertModel, setCurrentBertModel] = useState<string | null>(null);
  const { deleteBertModel } = useDeleteBertModel(projectSlug || null);

  // get model information from api
  const { model } = useModelInformations(
    projectSlug || null,
    currentBertModel || null,
    'bert',
    isComputing,
  );

  // compute model prediction
  const [batchSize, setBatchSize] = useState<number>(32);

  // form to rename
  const { renameBertModel } = useRenameBertModel(projectSlug || null);
  const {
    handleSubmit: handleSubmitRename,
    register: registerRename,
    reset: resetRename,
  } = useForm<renameModel>();

  const onSubmitRename: SubmitHandler<renameModel> = async (data) => {
    if (currentBertModel) {
      await renameBertModel(currentBertModel, data.new_name);
      resetRename();
    } else notify({ type: 'error', message: 'New name is void' });
  };

  const loss = model?.loss ? (model?.loss as unknown as LossData) : null;

  return (
    <Tabs id="bert" className="mt-1" defaultActiveKey="existing">
      <Tab eventKey="existing" title="Existing">
        <label htmlFor="selected-model">Existing models</label>
        <div className="d-flex align-items-center">
          <select
            id="selected-model"
            className="form-select"
            onChange={(e) => setCurrentBertModel(e.target.value)}
            value={currentBertModel || ''}
          >
            <option></option>
            {availableBertModels.map((e) => (
              <option key={e}>{e}</option>
            ))}
          </select>
          <button
            className="btn btn p-0"
            onClick={() => {
              if (currentBertModel) {
                deleteBertModel(currentBertModel);
                setCurrentBertModel(null);
              }
            }}
          >
            <MdOutlineDeleteOutline size={30} />
          </button>
        </div>

        {currentBertModel && (
          <div>
            {model && (
              <div>
                <details style={{ color: 'gray' }}>
                  <summary>
                    <span>Parameters of the model</span>
                  </summary>
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
                      className="m-2"
                      style={{ width: '50px' }}
                      value={batchSize}
                      onChange={(e) => setBatchSize(Number(e.target.value))}
                    />
                  </div>
                  <ModelParametersTab params={model.params as Record<string, unknown>} />
                  <details className="m-2">
                    <summary>Rename</summary>
                    <form onSubmit={handleSubmitRename(onSubmitRename)}>
                      <input
                        id="new_name"
                        className="form-control me-2 mt-2"
                        type="text"
                        placeholder="New name of the model"
                        {...registerRename('new_name')}
                      />
                      <button className="btn btn-primary me-2 mt-2">Rename</button>
                    </form>
                  </details>
                </details>
                {isComputing && (
                  <DisplayTrainingProcesses
                    projectSlug={projectSlug || null}
                    processes={project?.languagemodels.training}
                    displayStopButton={isComputing}
                  />
                )}

                <div className="mt-2">
                  <DisplayScores
                    title={'Internal validation'}
                    scores={model.scores.internalvalid_scores as MLStatisticsModel}
                    modelName={currentBertModel}
                  />
                </div>

                <div className="mt-2">
                  <LossChart loss={loss} />
                </div>
              </div>
            )}
          </div>
        )}
      </Tab>
      <Tab eventKey="new" title="New">
        <div className="explanations">
          The model will be trained on annotated data. A good practice is to have at least 50
          annotated elements. You can exclude elements with specific labels.{' '}
          <a className="problems m-2">
            <FaTools />
            <Tooltip anchorSelect=".problems" place="top">
              If the model doesn't train, the reason can be the limit of available GPU. Please try
              latter. If the problem persists, contact us.
            </Tooltip>
          </a>
        </div>
        <DisplayTrainingProcesses
          projectSlug={projectSlug || null}
          processes={project?.languagemodels.training}
          displayStopButton={isComputing}
        />

        <ModelCreationForm
          projectSlug={projectSlug || null}
          currentScheme={currentScheme || null}
          project={project || null}
          isComputing={isComputing}
        />
      </Tab>
    </Tabs>
  );
};
