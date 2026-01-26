import { FC, useEffect, useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';
import { useDeleteBertModel, useDeleteQuickModel, useModelInformations } from '../core/api';
import { useAppContext } from '../core/context';
import { useNotifications } from '../core/notifications';
import { sortDatesAsStrings } from '../core/utils';
import { MLStatisticsModel } from '../types';
import { DisplayScoresMenu } from './DisplayScoresMenu';
import { ModelsPillDisplay } from './ModelsPillDisplay';
import { ValidateButtons } from './ValidateButton';

export const ModelEvaluation: FC = () => {
  const { projectName: projectSlug } = useParams();
  const { notify } = useNotifications();
  const {
    appContext: { currentScheme, currentProject, isComputing },
  } = useAppContext();

  // quickmodel selector
  const availableQuickModels = useMemo(
    () => currentProject?.quickmodel.available[currentScheme || ''] || [],
    [currentProject?.quickmodel, currentScheme],
  );
  const [currentQuickModelName, setCurrentQuickModelName] = useState<string | null>(null);
  const { deleteQuickModel } = useDeleteQuickModel(projectSlug || null);

  // bertmodel selector
  const availableBertModels = useMemo(
    () => currentProject?.languagemodels.available[currentScheme || ''] || {},
    [currentProject?.languagemodels, currentScheme],
  );
  const [currentBertModel, setCurrentBertModel] = useState<string | null>(null);
  const { deleteBertModel } = useDeleteBertModel(projectSlug || null);

  // get model information from api
  const { model: bertModelInformations, reFetch: reFetchBertModelInformations } =
    useModelInformations(projectSlug || null, currentBertModel || null, 'bert', isComputing);
  const { model: quickModelInformations, reFetch: reFetchQuickModelInformations } =
    useModelInformations(projectSlug || null, currentQuickModelName || null, 'quick', isComputing);

  // refetch model informations when computation is done
  useEffect(() => {
    if (!isComputing) {
      reFetchBertModelInformations();
      reFetchQuickModelInformations();
    }
  }, [isComputing, reFetchBertModelInformations, reFetchQuickModelInformations]);

  // meta selector
  const [currentModel, setCurrentModel] = useState<{ name: string; kind: string } | null>(null);
  useEffect(() => {
    if (currentQuickModelName) {
      setCurrentModel({ name: currentQuickModelName, kind: 'quick' });
      setCurrentBertModel(null);
    }
  }, [currentQuickModelName]);
  useEffect(() => {
    if (currentBertModel) {
      setCurrentModel({ name: currentBertModel, kind: 'bert' });
      setCurrentQuickModelName(null);
    }
  }, [currentBertModel]);

  return (
    <div>
      {/* Display all the models */}
      <div>
        <span className="fw-semibold text-muted small">Quick Models</span>
        <ModelsPillDisplay
          modelNames={availableQuickModels
            .sort((quickModelA, quickModelB) =>
              sortDatesAsStrings(quickModelA?.time, quickModelB?.time, true),
            )
            .map((quickModel) => quickModel.name)}
          currentModelName={currentQuickModelName}
          setCurrentModelName={setCurrentQuickModelName}
          deleteModelFunction={deleteQuickModel}
        />
      </div>
      <div>
        <span className="fw-semibold text-muted small">BERT Models</span>
        <ModelsPillDisplay
          modelNames={Object.values(availableBertModels)
            .sort((bertModelA, bertModelB) =>
              sortDatesAsStrings(bertModelA?.time, bertModelB?.time, true),
            )
            .map((model) => (model ? model.name : ''))}
          currentModelName={currentBertModel}
          setCurrentModelName={setCurrentBertModel}
          deleteModelFunction={deleteBertModel}
        />
      </div>

      <hr className="my-4" />

      {quickModelInformations && currentModel && (
        <>
          <ValidateButtons
            modelName={currentQuickModelName}
            kind="quick"
            id="compute-validate"
            style={{ margin: '8px 0px', color: 'white' }}
          />

          <DisplayScoresMenu
            scores={quickModelInformations.scores as unknown as Record<string, MLStatisticsModel>}
            modelName={currentQuickModelName || ''}
            skip={['internalvalid_scores']}
            projectSlug={projectSlug || null}
          />
        </>
      )}

      {bertModelInformations && currentModel && (
        <>
          <ValidateButtons
            modelName={currentBertModel}
            kind="bert"
            id="compute-validate"
            style={{ margin: '8px 0px', color: 'white' }}
          />
          <DisplayScoresMenu
            scores={bertModelInformations.scores as unknown as Record<string, MLStatisticsModel>}
            modelName={currentQuickModelName || ''}
            skip={['internalvalid_scores']}
            projectSlug={projectSlug || null}
          />
        </>
      )}
    </div>
  );
};
